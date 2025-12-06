# NO_NETWORK Discipline - Network Isolation Framework

## Overview

The NO_NETWORK discipline provides comprehensive network isolation capabilities for MathLedger's CI environment. This framework enables all tests to run without external dependencies through mocks, stubs, and replay mechanisms.

## Architecture

### Core Components

1. **Database Mocks** (`backend/testing/no_network.py`)
   - Pattern-based cursor mocking
   - Schema-adaptive query responses
   - In-memory connection simulation

2. **Redis Mocks** (`backend/testing/no_network.py`)
   - In-memory queue operations
   - Full LPUSH/RPOP/LLEN support
   - Queue state management

3. **HTTP Replay Framework** (`backend/testing/no_network.py`)
   - Record/replay request/response pairs
   - Deterministic testing without network
   - JSON-based recording storage

4. **Network Sandbox** (`scripts/network-sandbox.py`)
   - Environment-aware isolation enforcement
   - Command execution wrapper
   - Validation and simulation modes

## Quick Start

### Enable NO_NETWORK Mode

```bash
export NO_NETWORK=true
```

### Run Tests in Sandbox

```bash
# Run all tests
python scripts/network-sandbox.py pytest tests/

# Run specific test file
python scripts/network-sandbox.py pytest tests/test_no_network.py

# Run with coverage
python scripts/network-sandbox.py coverage run -m pytest tests/
```

### Validate Configuration

```bash
python scripts/network-sandbox.py --validate
```

### Run Simulation

```bash
python scripts/network-sandbox.py --simulate
```

## Usage Examples

### Database Mocking

```python
from backend.testing.no_network import mock_psycopg_connect

# Create mock connection with custom rules
rules = [
    (lambda s, p: 'COUNT(*)' in s, lambda: ('one', (42,))),
    (lambda s, p: 'SELECT id' in s, lambda: ('all', [(1,), (2,), (3,)])),
]

connect = mock_psycopg_connect(rules)
conn = connect()

with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM proofs")
    count = cur.fetchone()[0]  # Returns 42
```

### Redis Mocking

```python
from backend.testing.no_network import MockRedis

# Create mock Redis client
redis = MockRedis()

# Queue operations
redis.lpush('ml:jobs', '{"statement": "p -> p"}')
redis.lpush('ml:jobs', '{"statement": "q -> q"}')

assert redis.llen('ml:jobs') == 2

job = redis.rpop('ml:jobs')
assert 'p -> p' in job
```

### HTTP Recording and Replay

```python
from backend.testing.no_network import HTTPRecorder, mock_requests_session

# Create recorder
recorder = HTTPRecorder('artifacts/no_network/recordings')

# Record request (run once with network)
recorder.record(
    'GET', 'http://api.example.com/data',
    None, 200, '{"result": "success"}',
    {'Content-Type': 'application/json'}
)

# Replay request (run in NO_NETWORK mode)
session = mock_requests_session(recorder)
response = session.get('http://api.example.com/data')
assert response.status_code == 200
assert response.json() == {"result": "success"}
```

### Network Sandbox Context

```python
from backend.testing.no_network import network_sandbox

with network_sandbox(strict=True) as sandbox:
    # All code here runs in isolated environment
    # Network attempts will raise exceptions
    pass
```

## Integration with Existing Tests

### Pattern-Based Mocking

The framework uses pattern-based mocking to adapt to schema variations:

```python
from backend.testing.no_network import MockConnection

rules = [
    # Schema introspection
    (lambda s, p: 'information_schema.columns' in s and 'proofs' in s,
     lambda: ('all', [('success', 'boolean'), ('created_at', 'timestamp')])),
    
    # Count queries
    (lambda s, p: 'COUNT(*) FROM proofs WHERE success' in s,
     lambda: ('one', (42,))),
    
    # Max queries
    (lambda s, p: 'MAX(block_number)' in s,
     lambda: ('one', (9,))),
]

conn = MockConnection(rules)
```

### Convenience Functions

```python
from backend.testing.no_network import (
    get_mock_db_connection,
    get_mock_redis,
    is_no_network_mode,
)

# Check if NO_NETWORK mode is enabled
if is_no_network_mode():
    conn = get_mock_db_connection()
    redis = get_mock_redis()
else:
    # Use real connections
    pass
```

## CI Integration

### GitHub Actions

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      NO_NETWORK: true
      DATABASE_URL: mock://testing
      REDIS_URL: mock://testing
    steps:
      - uses: actions/checkout@v4
      - name: Run tests in network sandbox
        run: |
          python scripts/network-sandbox.py pytest tests/
```

### Local Development

```bash
# Run tests without network
NO_NETWORK=true pytest tests/

# Or use the sandbox script
python scripts/network-sandbox.py pytest tests/
```

## Simulation Log

The simulation log (`artifacts/no_network/simulation.log`) provides ASCII-only output of all network isolation tests:

```
[2025-10-19T18:20:20.748761] [TEST] Simulation: Database mock
[2025-10-19T18:20:20.758012] [OK] Database mock query result: (0,)
[2025-10-19T18:20:20.758145] [OK] Database mock: PASS
[2025-10-19T18:20:20.758207] [TEST] Simulation: Redis mock
[2025-10-19T18:20:20.758316] [OK] Redis mock queue length: 2
[2025-10-19T18:20:20.758394] [OK] Redis mock pop result: job2
[2025-10-19T18:20:20.758451] [OK] Redis mock: PASS
[2025-10-19T18:20:20.758500] [TEST] Simulation: HTTP recorder
[2025-10-19T18:20:20.758944] [OK] HTTP recorder: PASS
[2025-10-19T18:20:20.759031] [TEST] Simulation: Network sandbox
[2025-10-19T18:20:20.759129] [OK] NO_NETWORK mode detected: true
[2025-10-19T18:20:20.759216] [OK] Network sandbox context: entered
[2025-10-19T18:20:20.759292] [OK] Network sandbox: PASS
[2025-10-19T18:20:20.759363] [INFO] Simulation complete
```

## Testing the Framework

Run the comprehensive test suite:

```bash
# Run NO_NETWORK framework tests
NO_NETWORK=true pytest tests/test_no_network.py -v

# Run with coverage
NO_NETWORK=true coverage run -m pytest tests/test_no_network.py
coverage report
```

## Best Practices

### 1. Always Set NO_NETWORK in CI

```yaml
env:
  NO_NETWORK: true
```

### 2. Use Pattern-Based Mocking for Schema Tolerance

```python
# Good: Adapts to schema changes
rules = [
    (lambda s, p: 'COUNT(*)' in s, lambda: ('one', (42,))),
]

# Bad: Hardcoded table/column names
rules = [
    (lambda s, p: s == 'SELECT COUNT(*) FROM proofs', lambda: ('one', (42,))),
]
```

### 3. Record HTTP Interactions Once

```bash
# Record with network (run once)
NO_NETWORK=false python record_interactions.py

# Replay without network (run in CI)
NO_NETWORK=true pytest tests/
```

### 4. Use Convenience Functions

```python
# Good: Uses convenience function
from backend.testing.no_network import get_mock_db_connection
conn = get_mock_db_connection()

# Also good: Direct instantiation with custom rules
from backend.testing.no_network import MockConnection
conn = MockConnection(custom_rules)
```

## Troubleshooting

### Issue: Tests fail with "NO_NETWORK mode not enabled"

**Solution**: Set the environment variable:
```bash
export NO_NETWORK=true
```

### Issue: Database queries return empty results

**Solution**: Add pattern matching rules for your queries:
```python
rules = [
    (lambda s, p: 'your_query_pattern' in s, lambda: ('one', (expected_result,))),
]
```

### Issue: HTTP replay not found

**Solution**: Record the interaction first:
```python
recorder = HTTPRecorder()
recorder.record('GET', 'http://api.example.com', None, 200, '{"data": "value"}')
```

## Future Enhancements

- Distributed proof node simulation
- Network latency simulation
- Failure injection for resilience testing
- Automatic recording mode detection
- Schema migration replay

## Hermetic Build Extensions

### Hermetic v1 - Deterministic Package Installs

Run hermetic validation and generate the replay manifest for full reproducibility.

```bash
# Full validation (ASCII-only output)
NO_NETWORK=true python scripts/hermetic-validate.py --full
```

Expected pass signal:

```
[PASS] NO_NETWORK HERMETIC: TRUE
```

Replay manifest location:

```
artifacts/no_network/replay_manifest.json
```

Core modules: `backend/testing/hermetic.py` (deterministic packages via HermeticPackageManager, ExternalAPIMockRegistry for PyPI/GitHub, ReplayLogComparator for cross-run comparison).

### Hermetic v2 - Full Reproducibility Validation

Extends hermetic v1 with RFC 8785 canonical JSON serialization, byte-identical replay log comparison, multi-lane validation, and fleet state archival on [PASS] ALL BLUE.

```bash
# Full v2 validation with all component tests
NO_NETWORK=true python scripts/hermetic-validate-v2.py --full
```

Expected pass signal:

```
[PASS] NO_NETWORK HERMETIC v2 TRUE
```

Artifacts generated:

```
artifacts/no_network/replay_manifest_v2.json  # RFC 8785 canonical manifest
artifacts/allblue/fleet_state.json            # Signed fleet state on ALL BLUE
```

Core modules: `backend/testing/hermetic_v2.py` (RFC8785Canonicalizer for byte-identical JSON, ByteIdenticalComparator for replay log comparison, MultiLaneValidator for all CI lanes, FleetStateArchiver for ALL BLUE freeze).

**Key Features:**
- RFC 8785 canonical JSON serialization for byte-identical comparison
- Multi-lane hermetic validation (dual-attestation, browsermcp, reasoning, test, uplift-omega, Compute Uplift Statistics)
- Fleet state archival with hash signing on [PASS] ALL BLUE
- Verifiable cognition chain advancement

## References

- `backend/testing/no_network.py` - Core framework implementation
- `scripts/network-sandbox.py` - Sandbox execution wrapper
- `tests/test_no_network.py` - Comprehensive test suite
- `artifacts/no_network/simulation.log` - Simulation output
