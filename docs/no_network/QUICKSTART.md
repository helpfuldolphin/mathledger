# NO_NETWORK Quick Start Guide

## 30-Second Setup

```bash
# 1. Enable NO_NETWORK mode
export NO_NETWORK=true

# 2. Run tests
python scripts/network-sandbox.py pytest tests/

# 3. Verify simulation
python scripts/network-sandbox.py --simulate
```

## Common Commands

### Run Tests

```bash
# All tests
python scripts/network-sandbox.py pytest tests/

# Specific test file
python scripts/network-sandbox.py pytest tests/test_no_network.py

# With verbose output
python scripts/network-sandbox.py pytest tests/ -v

# With coverage
python scripts/network-sandbox.py coverage run -m pytest tests/
```

### Validation

```bash
# Validate sandbox configuration
python scripts/network-sandbox.py --validate

# Run simulation tests
python scripts/network-sandbox.py --simulate

# Check simulation log
cat artifacts/no_network/simulation.log
```

### Development

```bash
# Run derive in sandbox
python scripts/network-sandbox.py python backend/axiom_engine/derive.py --smoke-pl

# Run API in sandbox
python scripts/network-sandbox.py uvicorn backend.orchestrator.app:app

# Run worker in sandbox
python scripts/network-sandbox.py python backend/worker.py
```

## Python API

### Database Mocking

```python
from backend.testing.no_network import get_mock_db_connection

conn = get_mock_db_connection()
with conn.cursor() as cur:
    cur.execute("SELECT COUNT(*) FROM proofs")
    count = cur.fetchone()[0]
```

### Redis Mocking

```python
from backend.testing.no_network import get_mock_redis

redis = get_mock_redis()
redis.lpush('ml:jobs', '{"statement": "p -> p"}')
job = redis.rpop('ml:jobs')
```

### HTTP Replay

```python
from backend.testing.no_network import HTTPRecorder, mock_requests_session

recorder = HTTPRecorder()
session = mock_requests_session(recorder)
response = session.get('http://api.example.com/data')
```

## CI Integration

### GitHub Actions

```yaml
jobs:
  test:
    runs-on: ubuntu-latest
    env:
      NO_NETWORK: true
    steps:
      - uses: actions/checkout@v4
      - run: python scripts/network-sandbox.py pytest tests/
```

### Pre-commit Hook

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: no-network-tests
        name: Run tests in NO_NETWORK mode
        entry: python scripts/network-sandbox.py pytest
        language: system
        pass_filenames: false
```

## Troubleshooting

### Tests fail with network errors

```bash
# Ensure NO_NETWORK is set
export NO_NETWORK=true

# Or use the sandbox script
python scripts/network-sandbox.py pytest tests/
```

### Mock returns wrong data

```python
# Add custom rules
from backend.testing.no_network import MockConnection

rules = [
    (lambda s, p: 'your_query' in s, lambda: ('one', (expected_result,))),
]
conn = MockConnection(rules)
```

### HTTP replay not found

```python
# Record first (with network)
recorder = HTTPRecorder()
recorder.record('GET', 'http://api.example.com', None, 200, '{"data": "value"}')

# Then replay (without network)
response = recorder.replay('GET', 'http://api.example.com')
```

## Next Steps

- Read full documentation: `docs/no_network/README.md`
- Review test examples: `tests/test_no_network.py`
- Check simulation log: `artifacts/no_network/simulation.log`
