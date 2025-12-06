# First Organism Telemetry & Grafana Dashboard Guide

This guide explains how to visualize the First Organism metrics emitted by the MathLedger Orchestrator.

## Metrics Endpoint

The metrics are exposed at:
`GET /metrics`

Response format (JSON):
```json
{
  "generated_at": "2023-10-27T10:00:00Z",
  "first_organism": {
    "runs_total": 42,
    "runs_completed": 40,
    "runs_failed": 2,
    "last_ht_hash": "a1b2c3d4...",
    "latency_seconds": 1.25,
    "last_abstention_rate": 0.137,
    "latency_buckets": {
      "0.1": 5,
      "0.5": 10,
      "1.0": 15,
      "+Inf": 42
    },
    "abstention_buckets": {
      "0.05": 2,
      "0.1": 8,
      "0.15": 18,
      "+Inf": 42
    }
  }
  // ... other system metrics
}
```

## Grafana Integration

To visualize these metrics in Grafana, we recommend using the **Infinity** data source plugin, which handles JSON endpoints natively.

### 1. Setup Data Source
1. Install the **Infinity** plugin in Grafana.
2. Add a new Data Source: `Infinity`.
3. Configuration:
   - **Type**: `JSON`
   - **URL**: `http://<orchestrator-host>:8000/metrics` (e.g. `http://localhost:8000/metrics`)
   - **Allowed Hosts**: Configure if necessary.

### 2. Dashboard Panels

#### A. First Organism Runs (Counter)
*Visualize the progression of test runs.*

- **Panel Type**: Stat or Time Series
- **Query**:
  - **Type**: JSON
  - **Root Selector**: `response`
  - **Columns**:
    - Selector: `first_organism.runs_total` -> Alias: `Total`
    - Selector: `first_organism.runs_completed` -> Alias: `Completed`
    - Selector: `first_organism.runs_failed` -> Alias: `Failed`

#### B. Latency Gauge
*Real-time view of the last run's latency.*

- **Panel Type**: Gauge
- **Query**:
  - Selector: `first_organism.latency_seconds`
  - Unit: `Seconds`

#### C. Latency Histogram (Bar Chart)
*Use the bucket counts to visualize percentile bands.*

- **Panel Type**: Bar Chart
- **Query**:
  - Selector: `first_organism.latency_buckets`
  - Format: Key-Value pair (Bucket Limit : Count)

#### D. Abstention Histogram
*Show how abstention rates are distributed relative to tolerance.*

- **Panel Type**: Bar Chart
- **Query**:
  - Selector: `first_organism.abstention_buckets`
  - Format: Bucket Upper Bound : Frequency

#### E. Latest Abstention Rate
*Gauge the last measured abstention rate to ensure it stays below policy tolerance.*

- **Panel Type**: Gauge
- **Query**:
  - Selector: `first_organism.last_abstention_rate`
  - Unit: `Percent` (multiply by 100 if desired)

#### F. Last Hₜ Hash
*The latest composite root seen by the RFL Runner.*

- **Panel Type**: Text / Table
- **Query**:
  - Selector: `first_organism.last_ht_hash`

## Troubleshooting

- **Missing Metrics**: Ensure the Redis instance is reachable by both the `RFLRunner` and the `Orchestrator`.
- **Zero Values**: Metrics are populated only after the first successful attestation run.

