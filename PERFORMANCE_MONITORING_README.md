# Performance Monitoring System
Cursor B - Performance & Memory Sanity Cartographer

This system maintains endpoint performance passports (JSON) with dual memory profilers (psutil + tracemalloc), deterministic outputs, and CI artifact uploads.

## Global Doctrine Compliance

- **ASCII-only logs**: No emojis in CI output
- **Deterministic comparison**: Via JSON hash
- **Mechanical honesty**: Status reflects API/test truth
- **Proof-or-Abstain**: If there is no verifiable proof, abstain

## Performance Requirements

The `/metrics` endpoint must meet these strict requirements:

- **Latency**: <10ms
- **Memory**: <10MB peak
- **Objects**: <1000 allocation

## Components

### 1. Enhanced /metrics Endpoint

The `/metrics` endpoint has been enhanced with dual memory profilers:

- **psutil**: Process-level memory monitoring
- **tracemalloc**: Python object-level memory tracking

```python
# Performance metrics are included in the response
{
  "proofs": {"success": 2850, "failure": 150},
  "block_count": 42,
  "max_depth": 8,
  # ... other metrics ...
  "performance": {
    "latency_ms": 2.345,
    "memory_delta_mb": 1.234,
    "object_delta": 45,
    "peak_memory_mb": 2.567,
    "initial_memory_mb": 15.432,
    "final_memory_mb": 16.666
  }
}
```

### 2. Performance Passport Generator

`scripts/generate-performance-passport.py` - Generates comprehensive performance passports with:

- Endpoint timing/memory metrics
- System information
- Performance thresholds
- Deterministic JSON hash for comparison

```bash
# Generate performance passport
python scripts/generate-performance-passport.py --api-url http://localhost:8000 --ci

# Compare with baseline
python scripts/generate-performance-passport.py --api-url http://localhost:8000 --baseline baseline_performance_passport.json
```

### 3. Baseline Passport Creator

`scripts/create-baseline-passport.py` - Creates baseline performance passports for comparison:

```bash
# Create baseline passport
python scripts/create-baseline-passport.py --api-url http://localhost:8000 --ci
```

### 4. CI Performance Check

`scripts/ci-performance-check.py` - Runs performance checks in CI environments:

```bash
# Run CI performance check
python scripts/ci-performance-check.py --api-url http://localhost:8000 --baseline baseline_performance_passport.json
```

### 5. Performance Requirements Validator

`scripts/validate-performance-requirements.py` - Validates that endpoints meet performance requirements:

```bash
# Validate performance requirements
python scripts/validate-performance-requirements.py --api-url http://localhost:8000 --iterations 100
```

### 6. Performance Formatter

`scripts/performance-formatter.py` - High-performance formatter for metrics data:

```bash
# Benchmark performance formatter
python scripts/performance-formatter.py --benchmark 1000
```

## Usage

### Local Development

1. **Start the API server**:
   ```bash
   make api
   ```

2. **Create baseline passport** (first time only):
   ```bash
   python scripts/create-baseline-passport.py --api-url http://localhost:8000
   ```

3. **Run performance validation**:
   ```bash
   python scripts/validate-performance-requirements.py --api-url http://localhost:8000 --iterations 50
   ```

4. **Generate performance passport**:
   ```bash
   python scripts/generate-performance-passport.py --api-url http://localhost:8000
   ```

### CI/CD Integration

The system includes GitHub Actions workflow (`.github/workflows/performance-check.yml`) that:

1. Starts database services
2. Runs migrations
3. Starts API server
4. Creates baseline passport (if not exists)
5. Runs performance checks
6. Uploads performance artifacts
7. Comments PR with performance results

### Performance Testing

Run the comprehensive performance tests:

```bash
# Run performance tests
uv run pytest tests/perf/test_core_endpoints_sanity.py -m perf_sanity -v

# Run with specific markers
uv run pytest -m "perf_sanity and not slow" -v
```

## Performance Passport Structure

```json
{
  "cartographer": "Cursor B - Performance & Memory Sanity Cartographer",
  "run_id": "20250102_143022",
  "session_id": "a1b2c3d4",
  "timestamp": "2025-01-02T14:30:22.123456",
  "performance_guarantee": "Even in sandbox mode, we never regress by more than 5%",
  "endpoints_profiled": ["metrics", "health", "blocks/latest"],
  "test_results": [
    {
      "endpoint": "metrics",
      "test_name": "metrics_variant_0",
      "latency_ms": 2.345,
      "memory_mb": 1.234,
      "objects": 45,
      "status": "PASS",
      "regression": false,
      "deterministic": true,
      "timestamp": "2025-01-02T14:30:22.234567"
    }
  ],
  "summary": {
    "total_tests": 18,
    "passed_tests": 18,
    "failed_tests": 0,
    "performance_regressions": 0,
    "memory_regressions": 0,
    "max_latency_ms": 5.678,
    "max_memory_mb": 3.456,
    "max_objects": 123,
    "deterministic_score": 100.0,
    "overall_status": "PASS"
  },
  "thresholds": {
    "max_latency_threshold_ms": 10.0,
    "max_memory_threshold_mb": 10.0,
    "max_objects_threshold": 1000,
    "regression_tolerance_percent": 5.0,
    "deterministic_threshold_percent": 95.0
  },
  "system_info": {
    "platform": "nt",
    "python_version": "3.11.9",
    "psutil_version": "7.1.0",
    "memory_total_gb": 15.72,
    "cpu_count": 20
  },
  "passport_hash": "a1b2c3d4e5f6..."
}
```

## Regression Detection

The system detects regressions by:

1. **Performance thresholds**: Exceeding latency, memory, or object limits
2. **Baseline comparison**: Comparing current passport hash with baseline
3. **Deterministic validation**: Ensuring consistent output across runs

## Troubleshooting

### Common Issues

1. **API not available**: Ensure the API server is running on the specified URL
2. **Performance regressions**: Check for recent changes that might affect performance
3. **Memory leaks**: Use the dual profilers to identify memory allocation patterns
4. **Deterministic failures**: Ensure all random elements are properly seeded

### Debugging

1. **Enable verbose logging**:
   ```bash
   python scripts/validate-performance-requirements.py --api-url http://localhost:8000 --iterations 10 -v
   ```

2. **Check performance passport**:
   ```bash
   cat performance_passport.json | jq '.summary'
   ```

3. **Compare with baseline**:
   ```bash
   diff baseline_performance_passport.json performance_passport.json
   ```

## Maintenance

### Updating Baselines

When performance characteristics change (e.g., new features, optimizations):

1. Update the baseline passport:
   ```bash
   python scripts/create-baseline-passport.py --api-url http://localhost:8000 --ci
   ```

2. Commit the new baseline:
   ```bash
   git add baseline_performance_passport.json
   git commit -m "Update performance baseline after optimization"
   ```

### Threshold Adjustments

If performance requirements change, update the thresholds in:

- `scripts/generate-performance-passport.py`
- `scripts/validate-performance-requirements.py`
- `tests/perf/test_core_endpoints_sanity.py`

## Grafana Integration

The First Organism metrics are exposed via the `/metrics` endpoint in JSON format, which can be ingested into Grafana using the JSON API data source or by scraping with a custom Prometheus exporter that parses JSON.

### Metrics Exposed

- `first_organism_runs_total`: Counter of First Organism runs started.
- `first_organism_last_ht_hash`: The last Hₜ (Composite Attestation Root) seen.
- `first_organism_latency_seconds`: Time from UI event to RFL update.

### Dashboard Setup (JSON API)

1.  **Install JSON API Data Source**: Ensure your Grafana instance has the JSON API plugin installed.
2.  **Add Data Source**: Configure a new data source pointing to `http://<your-api-host>/metrics`.
3.  **Create Dashboard**:
    *   **Total Runs**: Stat panel. Field: `first_organism.runs_total`.
    *   **Last Hₜ**: Text panel or Table. Field: `first_organism.last_ht_hash`.
    *   **Latency**: Gauge or Time Series (if historical data is stored). Field: `first_organism.latency_seconds`.

### Prometheus Adapter

If you prefer Prometheus:

1.  Run a sidecar exporter that queries `/metrics` and converts the JSON to Prometheus text format.
2.  Map:
    *   `first_organism.runs_total` -> `first_organism_runs_total` (Counter)
    *   `first_organism.last_ht_hash` -> `first_organism_info{hash="..."}` (Gauge=1)
    *   `first_organism.latency_seconds` -> `first_organism_latency_seconds` (Gauge or Histogram)

## Contributing

When contributing to the performance monitoring system:

1. **Follow the global doctrine**: ASCII-only logs, deterministic outputs
2. **Maintain performance requirements**: Ensure all changes meet the <10ms/<10MB/<1000 objects limits
3. **Update tests**: Add tests for new performance monitoring features
4. **Document changes**: Update this README when adding new components

## License

This performance monitoring system is part of the MathLedger project and follows the same license terms.