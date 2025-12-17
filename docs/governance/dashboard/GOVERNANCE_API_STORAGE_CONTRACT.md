# API & Storage Contract: Governance History Dashboard

**Objective:** Define a precise technical contract for the backend API and data storage/retention policies that will power the Governance History Dashboard.

---

## 1. REST API Definition

This API provides access to the historical governance log data.

### Endpoint: `GET /api/v1/governance/history`

Retrieves a list of governance log entries, ordered from newest to oldest by `timestamp`.

#### Query Parameters:

*   **`limit`** (integer, optional, default: 100): The maximum number of records to return. Used for "last N runs" queries.
*   **`start_time`** (string, optional): ISO 8601 timestamp. The beginning of the time window for the query.
*   **`end_time`** (string, optional): ISO 8601 timestamp. The end of the time window for the query.
*   **`layers`** (string, optional): A comma-separated list of layer names to include in the results (e.g., `P3,P4`). If omitted, all layers are returned.

#### Response Payload (200 OK):

The response is a JSON object containing the results and pagination information.

```json
{
  "data": [
    {
      "timestamp": "2025-12-10T10:05:00Z",
      "run_id": "run-a4b1f2",
      "layer": "P4",
      "governance_status": "WARNING",
      "metrics": {
        "delta_p": 0.15,
        "rsi": 78.9,
        "omega": 0.96,
        "divergence": 0.08,
        "quarantine_ratio": 0.05,
        "budget_invalid_percent": 0.8
      }
    },
    {
      "timestamp": "2025-12-10T10:00:00Z",
      "run_id": "run-a4b1c8",
      "layer": "P4",
      "governance_status": "STABLE",
      "metrics": {
        "delta_p": 0.05,
        "rsi": 65.2,
        "omega": 0.98,
        "divergence": 0.01,
        "quarantine_ratio": 0.0,
        "budget_invalid_percent": 0.1
      }
    }
  ],
  "pagination": {
    "limit": 100,
    "has_more": false
  }
}
```

---

## 2. Query Patterns

Here is how to translate common user queries into API requests.

*   **"Last 50 runs"**:
    `GET /api/v1/governance/history?limit=50`

*   **"All activity for layer P3 in the last 24 hours"**:
    `GET /api/v1/governance/history?layers=P3&start_time=2025-12-09T11:00:00Z&end_time=2025-12-10T11:00:00Z`

*   **"Last 1000 runs, but only for P4 and Substrate"**:
    `GET /api/v1/governance/history?limit=1000&layers=P4,Substrate`

---

## 3. Retention & Compaction Policy

This policy balances data granularity, query performance, and storage cost.

*   **Hot Tier (0-30 Days):**
    *   **Data:** Raw, full-precision metrics are stored.
    *   **Action:** No compaction or aggregation. Data is kept in a high-performance database for immediate querying.

*   **Warm Tier (31-180 Days):**
    *   **Data:** Raw metrics may be down-sampled to hourly or daily averages/min/max.
    *   **Action:** At 30 days, a background job aggregates the raw data. The original raw logs are compressed (`gzip`) and moved to cheaper, slower storage (e.g., archival blob storage). The aggregated data remains queryable.

*   **Cold Tier (>180 Days):**
    *   **Data:** Only daily summary statistics are retained in the queryable database.
    *   **Action:** At 180 days, the compressed raw logs from the warm tier are moved to long-term cold storage (e.g., AWS Glacier or equivalent).
    *   **Retention:** Full raw data is permanently deleted after 2 years. Aggregated daily summaries are retained for 7 years for compliance and long-term trend analysis.

---

## 4. Test Vectors (Sample I/O)

### Vector 1: Get the last run for layer P4

*   **Input Request:**
    `curl "http://localhost:8080/api/v1/governance/history?limit=1&layers=P4"`

*   **Expected Output:**
    ```json
    {
      "data": [
        {
          "timestamp": "2025-12-10T10:05:00Z",
          "run_id": "run-a4b1f2",
          "layer": "P4",
          "governance_status": "WARNING",
          "metrics": { "delta_p": 0.15, "rsi": 78.9, "omega": 0.96, "divergence": 0.08, "quarantine_ratio": 0.05, "budget_invalid_percent": 0.8 }
        }
      ],
      "pagination": { "limit": 1, "has_more": true }
    }
    ```

### Vector 2: Get all records within a specific 5-minute window

*   **Input Request:**
    `curl "http://localhost:8080/api/v1/governance/history?start_time=2025-12-10T09:58:00Z&end_time=2025-12-10T10:06:00Z"`

*   **Expected Output:**
    ```json
    {
      "data": [
        {
          "timestamp": "2025-12-10T10:05:00Z",
          "run_id": "run-a4b1f2",
          "layer": "P4",
          "governance_status": "WARNING",
          "metrics": { "delta_p": 0.15, "rsi": 78.9, "omega": 0.96, "divergence": 0.08, "quarantine_ratio": 0.05, "budget_invalid_percent": 0.8 }
        },
        {
          "timestamp": "2025-12-10T10:00:00Z",
          "run_id": "run-a4b1c8",
          "layer": "P4",
          "governance_status": "STABLE",
          "metrics": { "delta_p": 0.05, "rsi": 65.2, "omega": 0.98, "divergence": 0.01, "quarantine_ratio": 0.0, "budget_invalid_percent": 0.1 }
        }
      ],
      "pagination": { "limit": 100, "has_more": false }
    }
    ```
---

## 5. Query Performance Plan

This section outlines strategies to ensure the dashboard remains responsive as data volume grows.

### Indexing Strategy

To ensure efficient data retrieval from the `governance_events` table, the following indexes are critical. The exact `CREATE INDEX` syntax may vary slightly depending on the chosen database (e.g., PostgreSQL, MySQL, SQLite).

1.  **Primary Query Index (Composite):** The most common query pattern will be filtering by one or more `layers` and sorting by `timestamp`. A composite index is ideal for this.
    *   **Fields:** `(layer, timestamp DESC)`
    *   **Rationale:** Allows the database to quickly locate all records for a specific layer and retrieve them in the exact reverse chronological order required by the API, avoiding a costly sort operation.
    *   **SQL Example:** `CREATE INDEX idx_governance_events_layer_timestamp ON governance_events (layer, timestamp DESC);`

2.  **Time-Window Index:** For queries that filter by a time range but not a specific layer.
    *   **Fields:** `(timestamp DESC)`
    *   **Rationale:** Optimizes queries that use `start_time` and `end_time` without a `layers` filter.
    *   **SQL Example:** `CREATE INDEX idx_governance_events_timestamp ON governance_events (timestamp DESC);`

*(Note: In the current SQLite implementation, these indexes should be added to the `storage.py:initialize_db` function.)*

### Max Response Time SLAs

These are the target service level agreements (SLAs) for the `GET /api/v1/governance/history` endpoint under normal operating conditions with a dataset of up to 100 million records in the hot tier.

*   **p95 (95th percentile) Response Time:** `< 200ms` for standard queries (e.g., filtering by layer, 24-hour time window, limit <= 1000).
*   **p99 (99th percentile) Response Time:** `< 500ms` for more complex queries (e.g., spanning multiple layers over a 7-day window).
*   **Maximum Timeout:** The API gateway or load balancer should enforce a `30s` timeout to prevent catastrophic query runtimes.

### Load Test Stub



A load test stub will be created at `backend/dashboard/load_test.py` to simulate user traffic and verify that the API meets the defined SLAs. This stub will use `locust`, which should be installed from `requirements-dev.txt`.



---



## 6. Schema Migration Notes



The current implementation uses a simple, automatic migration strategy suitable for a lightweight service.



*   **Initialization:** On application startup, the `storage.initialize_db()` function is called.

*   **Idempotency:** This function uses `CREATE TABLE IF NOT EXISTS` and `CREATE INDEX IF NOT EXISTS` statements. This ensures that the schema setup is idempotent and can be safely run every time the application starts without causing errors or data loss.

*   **Process:** When the service is deployed or updated, the latest version of the schema (tables and indexes) is automatically applied upon the first startup. No manual migration steps are required for this initial version.
