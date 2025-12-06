# First Organism Telemetry Plan v2

**Document Version:** 2.0
**Author:** CLAUDE K (Telemetry Lead)
**Date:** 2025-11-30

---

> **STATUS: PHASE II — NOT YET IMPLEMENTED**
>
> This document describes a **future telemetry and analytics architecture** that has
> NOT been used in Evidence Pack v1. It is a design specification only.
>
> **Evidence Pack v1 uses only:**
> - FO Dyno Chart (1000-cycle baseline vs RFL abstention comparison)
> - Sealed attestation.json from closed-loop FO test
> - Raw logs in `results/fo_baseline_wide.jsonl` and `results/fo_rfl_wide.jsonl`
>
> The advanced analytics (anomaly detection, uplift CI computation, history export,
> time-series aggregation) described herein are **Phase II future work**.
>
> **RFL JSONL logs bypass this module; these analytics are Phase II only.**
>
> **Do NOT cite this document as evidence of production telemetry capabilities.**

---

## 1. Executive Summary

This document defines a **proposed** telemetry specification for the First Organism (FO) subsystem within MathLedger. It covers Redis key schemas, TTL policies, data retention, Dyno Chart aggregation guidance, RFL uplift mapping, and the canonical schema for `first_organism_history.json`.

**Phase I Reality:** The current implementation emits basic FO metrics to Redis via `first_organism_telemetry.py`. The Dyno Chart is generated from JSONL logs. No advanced analytics, anomaly detection, or automated history export is currently in production.

---

## 2. Redis Key Inventory

### 2.1 FO Metrics Keys (`ml:metrics:first_organism:*`)

| Key | Type | Description | Value Format |
|-----|------|-------------|--------------|
| `ml:metrics:first_organism:runs_total` | String | Cumulative run counter | Integer as string |
| `ml:metrics:first_organism:last_ht` | String | Latest H_t composite root (short) | 16-char hex string |
| `ml:metrics:first_organism:last_ht_full` | String | Latest H_t composite root (full) | 64-char hex string |
| `ml:metrics:first_organism:duration_seconds` | String | Last run duration | Float as string (6 decimals) |
| `ml:metrics:first_organism:latency_seconds` | String | Last run latency | Float as string |
| `ml:metrics:first_organism:last_abstentions` | String | Abstention count in last run | Integer as string |
| `ml:metrics:first_organism:last_run_timestamp` | String | Timestamp of last run | ISO 8601 string |
| `ml:metrics:first_organism:last_status` | String | Last run status | `"success"` or `"failure"` |
| `ml:metrics:first_organism:duration_history` | List | Rolling duration history | List of float strings |
| `ml:metrics:first_organism:abstention_history` | List | Rolling abstention history | List of integer strings |
| `ml:metrics:first_organism:success_history` | List | Rolling success/failure history | List of `"success"`/`"failure"` |
| `ml:metrics:first_organism:ht_history` | List | Rolling H_t hash history | List of 16-char hex strings |

### 2.2 FO Feedback Loop Keys (`ml:metrics:fo_feedback:*`)

| Key | Type | Description | Value Format |
|-----|------|-------------|--------------|
| `ml:metrics:fo_feedback:latest_decision` | String | Most recent feedback decision | JSON string |
| `ml:metrics:fo_feedback:decision_history` | List | Historical feedback decisions | List of JSON strings |
| `ml:metrics:fo_feedback:decisions_total` | String | Total feedback decisions counter | Integer as string |
| `ml:metrics:fo_feedback:throttle_count` | String | Cumulative throttle actions | Integer as string |
| `ml:metrics:fo_feedback:boost_count` | String | Cumulative boost actions | Integer as string |

### 2.3 Job Queue Keys

| Key | Type | Description | Value Format |
|-----|------|-------------|--------------|
| `ml:jobs` | List | Derivation verification job queue | List of JSON job objects |

---

## 3. TTL Policies

### 3.1 Current State: No TTLs

Currently, all FO metrics keys are stored **without TTL** (persistent). This is intentional for:

- **Counters** (`runs_total`, `decisions_total`, etc.): Cumulative metrics must persist indefinitely.
- **Scalar metrics** (`last_ht`, `last_status`, etc.): Represent current state, overwritten on each run.
- **History lists**: Self-limiting via `LTRIM` (max 20 entries for FO, max 100 for feedback).

### 3.2 Recommended TTL Strategy

| Key Category | Recommended TTL | Rationale |
|--------------|-----------------|-----------|
| `*:runs_total`, `*:decisions_total` | None (persistent) | Cumulative counters |
| `*:last_*` scalars | None (overwritten) | Always current state |
| `*:*_history` lists | None (bounded by LTRIM) | Self-limiting |
| `*:latest_decision` | 7 days (604800s) | Stale decisions become irrelevant |
| `ml:jobs` | None | Active queue, consumed promptly |

**Implementation Recommendation:** Add TTL to `latest_decision` key only, using `SETEX` or `EXPIRE` after `SET`.

### 3.3 Memory Budgeting

Estimated Redis memory per FO cycle:

| Component | Size Estimate |
|-----------|---------------|
| Scalar metrics (8 keys) | ~800 bytes |
| History lists (4 lists x 20 entries) | ~4 KB |
| Feedback decision (JSON) | ~1-2 KB |
| **Total per cycle** | ~6 KB |

At 100 cycles/day, daily growth: ~600 KB (negligible for Redis).

---

## 4. Data Retention Policy

### 4.1 Redis Retention

| Data Type | Retention | Enforcement |
|-----------|-----------|-------------|
| Scalar metrics | Indefinite (overwritten) | Natural overwrite |
| History lists | Last 20 entries | `LTRIM` after `LPUSH` |
| Feedback history | Last 100 entries | `LTRIM` after `LPUSH` |
| Job queue | Until consumed | `BLPOP` consumption |

### 4.2 Persistent Storage Retention

| Artifact | Location | Retention | Cleanup |
|----------|----------|-----------|---------|
| `first_organism_history.json` | `exports/` | 90 days | Nightly archival to `archives/` |
| Wide slice logs (`fo_*.jsonl`) | `results/` | 30 days | Manual cleanup after analysis |
| Dyno Chart images | `artifacts/figures/` | Indefinite | Manual curation |
| Lean job files | `backend/lean_proj/ML/Jobs/` | Last 500 files | Worker cleanup |

### 4.3 Database Retention (PostgreSQL)

FO metrics are NOT stored in PostgreSQL. All telemetry flows through Redis. Derived insights (block provenance, proof linkage) persist in:

- `proofs` table: Indefinite (immutable ledger)
- `blocks` table: Indefinite (blockchain headers)

---

## 5. Dyno Chart Aggregation Guidance

### 5.1 Data Sources

The Dyno Chart compares **Baseline** vs **RFL** abstention dynamics:

| Source | Path | Format |
|--------|------|--------|
| Baseline | `results/fo_baseline_wide.jsonl` | JSONL (cycle-indexed) |
| RFL | `results/fo_rfl_wide.jsonl` | JSONL (cycle-indexed) |

### 5.2 JSONL Schema (Wide Slice Logs)

```json
{
  "cycle": 0,
  "status": "success",
  "method": "lean-enabled",
  "verification_method": "lean",
  "abstention": false,
  "derivation": {
    "abstained": 3,
    "attempted": 100
  },
  "duration_seconds": 1.234567,
  "ht_hash": "abc123...",
  "timestamp": "2025-11-30T12:00:00Z"
}
```

### 5.3 Abstention Detection Logic

A cycle is counted as **abstained** if ANY of:

1. `status == "abstain"`
2. `method == "lean-disabled"` OR `verification_method == "lean-disabled"`
3. `abstention == true` OR `abstention == 1`
4. `derivation.abstained > 0`

### 5.4 Aggregation Pipeline

```
1. Load JSONL files → Parse cycle-indexed records
2. Sort by cycle number (ascending)
3. For each record:
   - Extract abstention flag using detection logic
   - Convert to binary: 1 (abstained) or 0 (not abstained)
4. Apply rolling window (default: 100 cycles)
   - Compute rolling mean of abstention binary
   - Result: abstention_rate per cycle (0.0 to 1.0)
5. Align baseline and RFL by cycle number
6. Plot both series on same axes
```

### 5.5 Chart Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window` | 100 | Rolling window size for moving average |
| `y_axis` | 0.0-1.0 | Abstention rate (fraction) |
| `x_axis` | Cycle index | Derivation cycle number |
| `baseline_color` | Blue | Baseline series color |
| `rfl_color` | Orange | RFL series color |

### 5.6 Generating the Dyno Chart

```bash
uv run python experiments/generate_dyno_chart.py \
    --baseline results/fo_baseline_wide.jsonl \
    --rfl results/fo_rfl_wide.jsonl \
    --window 100 \
    --output-name rfl_dyno_chart
```

Output: `artifacts/figures/rfl_dyno_chart.png`

---

## 6. FO Metrics → RFL Uplift Chart Mapping

> **PHASE II: This uplift computation is NOT used in Evidence Pack v1.**
> The Dyno Chart (Section 5) shows abstention rates only. Throughput uplift
> with confidence intervals is future work.

### 6.1 Overview

The RFL Uplift Chart would visualize the **throughput improvement** achieved by RFL-guided derivation over baseline. FO metrics would provide the input signals for this analysis.

### 6.2 Metric Mapping

| FO Metric | RFL Uplift Chart Usage |
|-----------|------------------------|
| `runs_total` | Sample size for confidence interval |
| `success_rate` | Primary health indicator |
| `abstention_count` | Inversely correlates with throughput |
| `duration_seconds` | Normalized throughput = 1 / duration |
| `health_status` | Validity gate (CRITICAL invalidates data) |

### 6.3 Uplift Computation

```
Throughput_baseline = 1 / mean(baseline_duration_history)
Throughput_rfl = 1 / mean(rfl_duration_history)

Uplift = (Throughput_rfl - Throughput_baseline) / Throughput_baseline * 100

Example:
  Baseline mean: 2.0s → Throughput: 0.5 proofs/sec
  RFL mean: 1.5s → Throughput: 0.667 proofs/sec
  Uplift: +33.4%
```

### 6.4 Confidence Interval Requirements

| Sample Size | CI Width | Recommendation |
|-------------|----------|----------------|
| < 10 | Wide (±30%) | Insufficient data |
| 10-30 | Moderate (±15%) | Early signal |
| 30-100 | Narrow (±5%) | Confident estimate |
| > 100 | Tight (±2%) | Production-ready |

### 6.5 Uplift Chart Data Schema

```json
{
  "timestamp": "2025-11-30T12:00:00Z",
  "baseline": {
    "mean_duration": 2.0,
    "throughput": 0.5,
    "sample_size": 50,
    "ci_95": [0.45, 0.55]
  },
  "rfl": {
    "mean_duration": 1.5,
    "throughput": 0.667,
    "sample_size": 50,
    "ci_95": [0.60, 0.73]
  },
  "uplift": {
    "value_percent": 33.4,
    "ci_95": [10.0, 56.8],
    "p_value": 0.023,
    "significant": true
  }
}
```

---

## 7. Schema: `first_organism_history.json`

> **PHASE II: This export format is NOT generated in Evidence Pack v1.**
> No `first_organism_history.json` file exists. This is a proposed schema
> for future automated history export.

### 7.1 File Location (Proposed)

```
exports/first_organism_history.json
```

### 7.2 Purpose (Future)

Canonical export of FO telemetry for:
- Archival and auditing
- External analysis tools
- Historical trend visualization
- Cross-session comparison

### 7.3 Top-Level Schema

```json
{
  "$schema": "https://mathledger.io/schemas/fo-history-v2.json",
  "version": "2.0",
  "generated_at": "2025-11-30T12:00:00Z",
  "generator": "fo_analytics.py",
  "summary": { ... },
  "current_state": { ... },
  "history": [ ... ],
  "feedback_log": [ ... ],
  "ht_verification": { ... },
  "metadata": { ... }
}
```

### 7.4 Section Schemas

#### 7.4.1 `summary`

```json
{
  "summary": {
    "runs_total": 150,
    "first_run_timestamp": "2025-11-01T00:00:00Z",
    "last_run_timestamp": "2025-11-30T12:00:00Z",
    "success_rate_overall": 92.0,
    "success_rate_recent": 95.0,
    "average_duration_seconds": 1.85,
    "median_duration_seconds": 1.72,
    "total_abstentions": 45,
    "average_abstentions_per_run": 0.3,
    "health_trend": "stable",
    "data_quality_score": 0.98
  }
}
```

#### 7.4.2 `current_state`

```json
{
  "current_state": {
    "runs_total": 150,
    "last_ht_hash": "abc123def456...",
    "last_ht_full": "abc123def456789...",
    "last_duration_seconds": 1.65,
    "last_abstentions": 0,
    "last_run_timestamp": "2025-11-30T12:00:00Z",
    "last_status": "success",
    "health_status": "ALIVE",
    "duration_trend": "down",
    "abstention_trend": "flat",
    "success_trend": "up"
  }
}
```

#### 7.4.3 `history`

Array of run records (newest first, up to 1000 entries):

```json
{
  "history": [
    {
      "run_index": 150,
      "timestamp": "2025-11-30T12:00:00Z",
      "duration_seconds": 1.65,
      "abstention_count": 0,
      "status": "success",
      "ht_hash": "abc123...",
      "ht_full": "abc123def456789...",
      "health_status": "ALIVE",
      "success_rate_at_time": 92.0,
      "duration_delta": -0.15,
      "abstention_delta": 0,
      "metadata": {}
    }
  ]
}
```

#### 7.4.4 `feedback_log`

Array of feedback decisions (newest first, up to 100 entries):

```json
{
  "feedback_log": [
    {
      "timestamp": "2025-11-30T12:00:00Z",
      "decision_hash": "abc123...",
      "action_taken": "none",
      "signal": {
        "success_rate": 92.0,
        "health_status": "ALIVE",
        "duration_trend": "down",
        "abstention_trend": "flat",
        "should_throttle": false,
        "should_boost": false,
        "confidence": 1.0,
        "intensity_multiplier": 1.0
      },
      "adjusted_params": {},
      "experiment_id": "rfl-exp-001"
    }
  ]
}
```

#### 7.4.5 `ht_verification`

```json
{
  "ht_verification": {
    "verified": true,
    "valid_count": 20,
    "total_count": 20,
    "unique_count": 20,
    "uniqueness_ratio": 1.0,
    "consistency_check": true,
    "truncation_check": true,
    "last_verified_ht": "abc123...",
    "last_verified_ht_full": "abc123def456789...",
    "anomalies": []
  }
}
```

#### 7.4.6 `metadata`

```json
{
  "metadata": {
    "redis_url": "redis://localhost:6379/0",
    "history_window": 20,
    "feedback_history_max": 100,
    "export_history_max": 1000,
    "schema_version": "2.0",
    "generator_version": "1.0.0",
    "environment": "production"
  }
}
```

### 7.5 Complete Example

```json
{
  "$schema": "https://mathledger.io/schemas/fo-history-v2.json",
  "version": "2.0",
  "generated_at": "2025-11-30T12:00:00Z",
  "generator": "fo_analytics.py",
  "summary": {
    "runs_total": 150,
    "first_run_timestamp": "2025-11-01T00:00:00Z",
    "last_run_timestamp": "2025-11-30T12:00:00Z",
    "success_rate_overall": 92.0,
    "success_rate_recent": 95.0,
    "average_duration_seconds": 1.85,
    "median_duration_seconds": 1.72,
    "total_abstentions": 45,
    "average_abstentions_per_run": 0.3,
    "health_trend": "stable",
    "data_quality_score": 0.98
  },
  "current_state": {
    "runs_total": 150,
    "last_ht_hash": "abc123def456",
    "last_ht_full": "abc123def456789012345678901234567890123456789012345678901234",
    "last_duration_seconds": 1.65,
    "last_abstentions": 0,
    "last_run_timestamp": "2025-11-30T12:00:00Z",
    "last_status": "success",
    "health_status": "ALIVE",
    "duration_trend": "down",
    "abstention_trend": "flat",
    "success_trend": "up"
  },
  "history": [],
  "feedback_log": [],
  "ht_verification": {
    "verified": true,
    "valid_count": 20,
    "total_count": 20,
    "unique_count": 20,
    "uniqueness_ratio": 1.0,
    "consistency_check": true,
    "truncation_check": true
  },
  "metadata": {
    "redis_url": "redis://localhost:6379/0",
    "history_window": 20,
    "feedback_history_max": 100,
    "export_history_max": 1000,
    "schema_version": "2.0"
  }
}
```

---

## 8. Analytics Module Specification

> **PHASE II: The `fo_analytics.py` module is a skeleton only.**
> It is NOT wired into the live metrics path. No anomaly detection,
> uplift computation, or automated export runs in production.
> Evidence Pack v1 does NOT use this module.

### 8.1 Module: `backend/metrics/fo_analytics.py`

**Status:** Skeleton implementation. Not integrated. Not tested in production.

Proposed higher-order analytics for FO telemetry, including:

- Time-series aggregation
- Anomaly detection
- Cross-correlation analysis
- Export generation
- Uplift computation

### 8.2 Core Classes

| Class | Purpose |
|-------|---------|
| `FOAnalytics` | Main analytics engine |
| `FOTimeSeriesAggregator` | Rolling window computations |
| `FOAnomalyDetector` | Z-score and threshold-based anomaly detection |
| `FOUpliftCalculator` | Baseline vs RFL uplift statistics |
| `FOHistoryExporter` | Generate `first_organism_history.json` |

### 8.3 Key Methods

| Method | Description |
|--------|-------------|
| `aggregate_duration_series()` | Compute rolling mean/median/std for durations |
| `aggregate_abstention_series()` | Compute rolling abstention metrics |
| `detect_anomalies()` | Identify outliers using Z-score (threshold: 2.5) |
| `compute_uplift()` | Calculate throughput uplift with CI |
| `export_history()` | Generate canonical JSON export |
| `load_wide_slice_logs()` | Parse JSONL files for Dyno Chart |
| `align_baseline_rfl()` | Align baseline and RFL by cycle |

---

## 9. Implementation Checklist (Phase II)

> **All items below are FUTURE WORK. None are complete.**

- [ ] Add TTL to `ml:metrics:fo_feedback:latest_decision` (7 days)
- [x] Create `backend/metrics/fo_analytics.py` skeleton *(skeleton only, not integrated)*
- [ ] Implement `FOHistoryExporter` class *(skeleton exists, not production-ready)*
- [ ] Add `export_history()` to nightly script
- [ ] Create JSON schema file at `schemas/fo-history-v2.json`
- [ ] Update `__init__.py` to export analytics classes
- [ ] Add unit tests for analytics functions
- [ ] Document Dyno Chart generation in `README_ops.md`
- [ ] Wire analytics into live metrics path
- [ ] Validate against real 1000-cycle runs

---

## 10. Appendices

### A. Redis Key Pattern Reference

```
ml:metrics:first_organism:*     # FO vital signs
ml:metrics:fo_feedback:*        # Feedback loop decisions
ml:jobs                         # Derivation job queue
```

### B. Health Status Thresholds

| Status | Condition |
|--------|-----------|
| `ALIVE` | `last_status == "success"` AND `success_rate >= 80%` |
| `DEGRADED` | `success_rate >= 50%` |
| `CRITICAL` | `success_rate < 50%` |
| `UNKNOWN` | `runs_total == 0` |

### C. Trend Epsilon Values

| Metric | Epsilon | Meaning |
|--------|---------|---------|
| Duration | 0.01s | < 10ms change = flat |
| Abstention | 0.01 | < 1% change = flat |
| Success rate | 0.10 | < 10% change = flat |

---

## 11. Phase II: RFL Uplift Metrics Schema

> **STATUS: PHASE II ONLY — NO SUCH METRICS EXIST IN PHASE I**
>
> This section defines a proposed metrics schema for observing RFL uplift.
> Phase I treats all RFL logs as file-only (`results/fo_*.jsonl`) with no
> analytics integration. The metrics below do not exist and are not collected.

### 11.1 Proposed Metrics

| Metric Name | Labels | Type | Description |
|-------------|--------|------|-------------|
| `rfl_abstention_rate` | `slice`, `policy_version` | Gauge (0.0–1.0) | Fraction of cycles where RFL abstained |
| `baseline_abstention_rate` | `slice` | Gauge (0.0–1.0) | Fraction of cycles where baseline abstained |
| `rfl_uplift_delta` | `slice` | Gauge | `baseline_abstention_rate - rfl_abstention_rate` (positive = RFL better) |
| `rfl_throughput` | `slice`, `policy_version` | Gauge | Proofs per second under RFL policy |
| `baseline_throughput` | `slice` | Gauge | Proofs per second under baseline policy |
| `rfl_throughput_uplift_pct` | `slice` | Gauge | `(rfl - baseline) / baseline * 100` |

### 11.2 Label Definitions

| Label | Values | Description |
|-------|--------|-------------|
| `slice` | `"narrow"`, `"wide"`, `"full"` | Derivation slice scope |
| `policy_version` | `"v1"`, `"v2"`, ... | RFL policy version identifier |

### 11.3 Aggregation Strategy

| Aggregation | Window | Method | Notes |
|-------------|--------|--------|-------|
| Per-cycle | 1 cycle | Raw value | Granular, high cardinality |
| Rolling | 100 cycles | Moving average | Matches Dyno Chart window |
| Epoch | Full run | Mean over all cycles | Summary statistic |

**Recommended default:** Rolling 100-cycle window for real-time dashboards; epoch aggregation for sealed reports.

### 11.4 Storage Options (Proposed)

| Option | Location | Format | Pros | Cons |
|--------|----------|--------|------|------|
| Redis time-series | `ml:metrics:rfl:*` | Sorted set with timestamps | Fast reads, TTL support | Memory-bound |
| PostgreSQL table | `rfl_metrics` | Rows with `(timestamp, slice, metric, value)` | Durable, queryable | Schema migration required |
| JSONL append-log | `results/rfl_metrics.jsonl` | One JSON object per cycle | Simple, no infra | No indexing |

**Phase II recommendation:** PostgreSQL table `rfl_metrics` with indexed `(slice, timestamp)` for efficient range queries.

### 11.5 Proposed PostgreSQL Schema

```sql
-- PHASE II ONLY: This table does not exist in Phase I

CREATE TABLE rfl_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    slice VARCHAR(20) NOT NULL,           -- 'narrow', 'wide', 'full'
    policy_version VARCHAR(20),           -- 'v1', 'v2', etc.
    metric_name VARCHAR(50) NOT NULL,     -- e.g., 'rfl_abstention_rate'
    value DOUBLE PRECISION NOT NULL,
    window_size INTEGER,                  -- NULL for raw, 100 for rolling
    run_id VARCHAR(64),                   -- Links to RFL experiment run
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rfl_metrics_slice_ts ON rfl_metrics (slice, timestamp);
CREATE INDEX idx_rfl_metrics_metric ON rfl_metrics (metric_name, timestamp);
```

### 11.6 Integration Contract (Design Only)

```
Phase I (Current):
  RFL Runner → JSONL logs → Dyno Chart generator → PNG artifact
  (No metrics collection, no database writes, no real-time aggregation)

Phase II (Proposed):
  RFL Runner → JSONL logs ─┬→ Dyno Chart generator → PNG artifact
                           │
                           └→ RFL Metrics Collector → rfl_metrics table
                                                    ↓
                           FOAnalytics.compute_uplift() ← reads from table
                                                    ↓
                           Dashboard / Alerts / Reports
```

### 11.7 Phase I vs Phase II Boundary

| Capability | Phase I | Phase II |
|------------|---------|----------|
| JSONL log generation | ✓ Exists | ✓ Continues |
| Dyno Chart from JSONL | ✓ Exists | ✓ Continues |
| `rfl_metrics` table | ✗ Does not exist | Proposed |
| Real-time abstention rate | ✗ Not collected | Proposed |
| Uplift delta computation | ✗ Manual only | Automated |
| `FOAnalytics` integration | ✗ Not wired | Proposed |

---

## 12. PHASE II — U2 Uplift Metric Families

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              NOT IMPLEMENTED — FUTURE PHASE II                                ║
║                                                                               ║
║  This section defines the core metric families for U2 (Uplift v2)             ║
║  experiments. These metrics are designed for slice-specific uplift            ║
║  measurement with confidence intervals.                                       ║
║                                                                               ║
║  ⚠  NO COLLECTION, COMPUTATION, OR STORAGE OF THESE METRICS EXISTS.          ║
║                                                                               ║
║              PHASE II — NOT RUN IN PHASE I                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 12.1 Core Metric Families

The U2 runner requires four core metric families to measure asymmetric uplift across environments:

| Metric Family | Symbol | Type | Range | Description |
|---------------|--------|------|-------|-------------|
| **Baseline Success Probability** | `p_base` | Gauge | [0.0, 1.0] | Probability of successful proof derivation under baseline (random) policy |
| **RFL Success Probability** | `p_rfl` | Gauge | [0.0, 1.0] | Probability of successful proof derivation under RFL policy |
| **Uplift Delta** | `uplift_delta` | Gauge | [-1.0, 1.0] | `p_rfl - p_base`; positive indicates RFL outperforms baseline |
| **Confidence Interval** | `CI` | Composite | [-1.0, 1.0] | 95% CI bounds (`ci_lower`, `ci_upper`) for `uplift_delta` |

### 12.1.1 Metric Family Hierarchy

```
                    ┌─────────────────────────────────────┐
                    │         U2 Metric Families          │
                    │    (NOT IMPLEMENTED - PHASE II)     │
                    └─────────────────┬───────────────────┘
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         │                            │                            │
         ▼                            ▼                            ▼
   ┌───────────┐              ┌───────────────┐            ┌──────────────┐
   │  p_base   │              │    p_rfl      │            │ uplift_delta │
   │  (Gauge)  │              │   (Gauge)     │            │   (Gauge)    │
   └─────┬─────┘              └───────┬───────┘            └──────┬───────┘
         │                            │                           │
         └────────────────────────────┴───────────────────────────┘
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │         CI          │
                           │    (Composite)      │
                           │ ci_lower, ci_upper  │
                           │ p_value, significant│
                           └─────────────────────┘
```

### 12.1.2 Metric Cardinality

| Metric | Per-Cycle | Per-Epoch | Per-Slice | Total (4 envs × 1000 cycles) |
|--------|-----------|-----------|-----------|------------------------------|
| `p_base` | ✗ | ✓ | ✓ | 4 values |
| `p_rfl` | ✗ | ✓ | ✓ | 4 values |
| `uplift_delta` | ✗ | ✓ | ✓ | 4 values |
| `CI` | ✗ | ✓ | ✓ | 4 composite records |

### 12.2 Metric Definitions

#### 12.2.1 `p_base` — Baseline Success Probability

```
p_base = (successful_cycles_baseline) / (total_cycles_baseline)

Where:
  successful_cycle := cycle.status == "success" AND NOT abstained(cycle)
  abstained(cycle) := cycle.abstention == true OR cycle.derivation.abstained > 0
```

**Computation window:** Per-epoch (full run), or rolling 100-cycle window.

**Example:**
```json
{
  "metric": "p_base",
  "slice": "U2_env_A",
  "value": 0.72,
  "sample_size": 1000,
  "window": "epoch",
  "timestamp": "2025-12-01T00:00:00Z"
}
```

#### 12.2.2 `p_rfl` — RFL Success Probability

```
p_rfl = (successful_cycles_rfl) / (total_cycles_rfl)
```

Same success/abstention definitions as `p_base`, applied to RFL-guided runs.

**Example:**
```json
{
  "metric": "p_rfl",
  "slice": "U2_env_A",
  "policy_version": "rfl_v2",
  "value": 0.85,
  "sample_size": 1000,
  "window": "epoch",
  "timestamp": "2025-12-01T00:00:00Z"
}
```

#### 12.2.3 `uplift_delta` — Absolute Uplift

```
uplift_delta = p_rfl - p_base
```

| Value Range | Interpretation |
|-------------|----------------|
| `> 0` | RFL outperforms baseline |
| `= 0` | No difference (null result) |
| `< 0` | Baseline outperforms RFL (negative uplift) |

**Example:**
```json
{
  "metric": "uplift_delta",
  "slice": "U2_env_A",
  "value": 0.13,
  "p_base": 0.72,
  "p_rfl": 0.85,
  "timestamp": "2025-12-01T00:00:00Z"
}
```

#### 12.2.4 `CI` — Confidence Interval for Uplift

95% confidence interval for `uplift_delta` using Wald interval for difference of proportions:

```
SE = sqrt(p_base * (1 - p_base) / n_base + p_rfl * (1 - p_rfl) / n_rfl)
CI_95 = uplift_delta ± 1.96 * SE
```

**Significance determination:**
- If `ci_lower > 0`: Statistically significant positive uplift
- If `ci_upper < 0`: Statistically significant negative uplift
- If CI spans 0: No statistically significant difference

**Example:**
```json
{
  "metric": "uplift_ci",
  "slice": "U2_env_A",
  "uplift_delta": 0.13,
  "ci_lower": 0.08,
  "ci_upper": 0.18,
  "ci_level": 0.95,
  "significant": true,
  "p_value": 0.0001,
  "timestamp": "2025-12-01T00:00:00Z"
}
```

### 12.3 Slice-Specific Success Metrics

Each U2 environment defines its own success criteria:

| Environment | Slice ID | Success Criterion | Target `p_rfl` |
|-------------|----------|-------------------|----------------|
| U2 Env A | `U2_env_A` | Proof completion without abstention | ≥ 0.80 |
| U2 Env B | `U2_env_B` | Throughput above threshold | ≥ 0.75 |
| U2 Env C | `U2_env_C` | Depth advancement achieved | ≥ 0.70 |
| U2 Env D | `U2_env_D` | Novel theorem discovery | ≥ 0.50 |

**Per-slice thresholds are defined in `PREREG_UPLIFT_U2.yaml`** and must not be modified after experiment preregistration.

---

## 13. PHASE II — Label Schema

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              NOT IMPLEMENTED — FUTURE PHASE II                                ║
║                                                                               ║
║  Labels provide dimensional indexing for metrics. This schema ensures         ║
║  consistent tagging across all U2 telemetry.                                  ║
║                                                                               ║
║  ⚠  NOT IMPLEMENTED IN PHASE I                                                ║
║                                                                               ║
║              PHASE II — NOT RUN IN PHASE I                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 13.1 Required Labels

| Label | Type | Cardinality | Values | Description |
|-------|------|-------------|--------|-------------|
| `slice` | string | 4 | `U2_env_A`, `U2_env_B`, `U2_env_C`, `U2_env_D` | Asymmetric uplift environment identifier |
| `policy` | string | 2 | `baseline`, `rfl` | Policy type under test |
| `policy_version` | string | Open | `v1`, `v2`, `rfl_v2_exp1`, ... | Specific policy version identifier |
| `run_id` | string | Open | UUID | Unique experiment run identifier |
| `window` | string | 3 | `cycle`, `rolling`, `epoch` | Aggregation window type |

### 13.2 Optional Labels

| Label | Type | Default | Description |
|-------|------|---------|-------------|
| `system_id` | integer | `1` | Logical system ID (1=PL, 2=FOL) |
| `depth_max` | integer | `4` | Maximum derivation depth |
| `breadth_max` | integer | `500` | Maximum breadth per step |
| `seed` | integer | `null` | RNG seed for reproducibility |
| `preregistration_id` | string | `null` | Link to preregistration document |

### 13.3 Label Formatting Rules

```
1. Snake_case for multi-word labels: `policy_version`, not `policyVersion`
2. Lowercase values except for slice IDs: `baseline`, not `Baseline`
3. Slice IDs use format: `U2_env_<letter>` (uppercase letter)
4. Version strings: `v<number>` or descriptive `rfl_v2_<experiment>`
5. UUIDs: lowercase hex with hyphens: `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
```

### 13.4 Label Validation Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["slice", "policy", "run_id"],
  "properties": {
    "slice": {
      "type": "string",
      "enum": ["U2_env_A", "U2_env_B", "U2_env_C", "U2_env_D"]
    },
    "policy": {
      "type": "string",
      "enum": ["baseline", "rfl"]
    },
    "policy_version": {
      "type": "string",
      "pattern": "^v[0-9]+|rfl_v[0-9]+.*$"
    },
    "run_id": {
      "type": "string",
      "format": "uuid"
    },
    "window": {
      "type": "string",
      "enum": ["cycle", "rolling", "epoch"]
    }
  }
}
```

---

## 14. PHASE II — Storage Model (JSONL Ingestion)

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              NOT IMPLEMENTED — FUTURE PHASE II                                ║
║                                                                               ║
║  This section defines the JSONL-based storage model for U2 metrics            ║
║  ingestion. All metrics are appended to per-slice log files, enabling         ║
║  offline analysis and batch processing.                                       ║
║                                                                               ║
║  ⚠  NO JSONL INGESTION PIPELINE EXISTS IN PHASE I                             ║
║                                                                               ║
║              PHASE II — NOT RUN IN PHASE I                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 14.1 File Layout

```
results/
├── u2_metrics/
│   ├── U2_env_A/
│   │   ├── baseline_<run_id>.jsonl
│   │   └── rfl_<run_id>.jsonl
│   ├── U2_env_B/
│   │   ├── baseline_<run_id>.jsonl
│   │   └── rfl_<run_id>.jsonl
│   ├── U2_env_C/
│   │   ├── baseline_<run_id>.jsonl
│   │   └── rfl_<run_id>.jsonl
│   └── U2_env_D/
│       ├── baseline_<run_id>.jsonl
│       └── rfl_<run_id>.jsonl
```

### 14.2 JSONL Record Schema (Per-Cycle)

Each line in the JSONL file represents one derivation cycle:

```json
{
  "schema_version": "2.0",
  "record_type": "cycle",
  "timestamp": "2025-12-01T00:00:00.123456Z",
  "labels": {
    "slice": "U2_env_A",
    "policy": "rfl",
    "policy_version": "rfl_v2",
    "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "window": "cycle"
  },
  "cycle": {
    "index": 42,
    "status": "success",
    "abstained": false,
    "duration_seconds": 1.234567,
    "derivations_attempted": 100,
    "derivations_abstained": 3,
    "proofs_verified": 97
  },
  "ht": {
    "hash_short": "abc123def456",
    "hash_full": "abc123def456789012345678901234567890123456789012345678901234"
  }
}
```

### 14.3 JSONL Record Schema (Epoch Summary)

Appended at end of each run:

```json
{
  "schema_version": "2.0",
  "record_type": "epoch_summary",
  "timestamp": "2025-12-01T01:00:00Z",
  "labels": {
    "slice": "U2_env_A",
    "policy": "rfl",
    "policy_version": "rfl_v2",
    "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "window": "epoch"
  },
  "summary": {
    "total_cycles": 1000,
    "successful_cycles": 850,
    "abstained_cycles": 50,
    "p_success": 0.85,
    "mean_duration_seconds": 1.45,
    "median_duration_seconds": 1.32,
    "total_proofs": 85000
  },
  "metrics": {
    "p_rfl": 0.85,
    "sample_size": 1000
  }
}
```

### 14.4 JSONL Record Schema (Uplift Computation)

Generated after paired baseline + RFL runs:

```json
{
  "schema_version": "2.0",
  "record_type": "uplift",
  "timestamp": "2025-12-01T02:00:00Z",
  "labels": {
    "slice": "U2_env_A",
    "baseline_run_id": "aaaa-bbbb-cccc-dddd",
    "rfl_run_id": "eeee-ffff-gggg-hhhh",
    "window": "epoch"
  },
  "uplift": {
    "p_base": 0.72,
    "p_rfl": 0.85,
    "uplift_delta": 0.13,
    "ci_lower": 0.08,
    "ci_upper": 0.18,
    "ci_level": 0.95,
    "significant": true,
    "p_value": 0.0001
  },
  "sample_sizes": {
    "baseline": 1000,
    "rfl": 1000
  }
}
```

### 14.5 Ingestion Pipeline (Proposed)

```
                    ┌──────────────────────────────────────┐
                    │  U2 Runner (baseline or rfl mode)    │
                    └──────────────────┬───────────────────┘
                                       │ append per-cycle
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  JSONL Log File (per-slice, per-run) │
                    │  results/u2_metrics/<slice>/*.jsonl  │
                    └──────────────────┬───────────────────┘
                                       │ batch ingest
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  JSONL Ingester (Phase II module)    │
                    │  - Validates schema                  │
                    │  - Computes epoch summaries          │
                    │  - Pairs baseline + RFL runs         │
                    │  - Computes uplift + CI              │
                    └──────────────────┬───────────────────┘
                                       │ write
                                       ▼
                    ┌──────────────────────────────────────┐
                    │  PostgreSQL rfl_metrics table        │
                    │  (optional: for dashboard queries)   │
                    └──────────────────────────────────────┘
```

### 14.6 Ingestion API (Proposed)

```python
# PHASE II — NOT IMPLEMENTED

class U2MetricsIngester:
    """
    Ingests JSONL metric logs and computes uplift statistics.

    NOT IMPLEMENTED — FUTURE PHASE II
    """

    def ingest_run(self, jsonl_path: Path) -> EpochSummary:
        """Parse JSONL file and compute epoch summary."""
        ...

    def compute_uplift(
        self,
        baseline_summary: EpochSummary,
        rfl_summary: EpochSummary
    ) -> UpliftResult:
        """Compute uplift delta with 95% CI."""
        ...

    def validate_record(self, record: dict) -> bool:
        """Validate record against schema."""
        ...

    def pair_runs(
        self,
        slice_id: str
    ) -> List[Tuple[EpochSummary, EpochSummary]]:
        """Find matching baseline + RFL run pairs for a slice."""
        ...
```

### 14.7 Storage Retention

| Artifact | Location | Retention | Cleanup |
|----------|----------|-----------|---------|
| Per-cycle JSONL | `results/u2_metrics/<slice>/` | 90 days | Archive to `archives/u2_metrics/` |
| Epoch summary JSONL | `results/u2_metrics/<slice>/` | 180 days | Archive after analysis |
| Uplift records | `results/u2_metrics/<slice>/` | Indefinite | Manual curation |
| PostgreSQL rows | `rfl_metrics` table | 1 year | Partition pruning |

### 14.8 Schema Versioning

| Version | Changes | Date |
|---------|---------|------|
| `1.0` | Initial schema (Phase I dyno chart) | 2025-11-30 |
| `2.0` | Added U2 labels, uplift records, CI | 2025-12-01 |

**Forward compatibility:** Ingester ignores unknown fields.
**Backward compatibility:** Missing optional fields use defaults.

### 14.9 Proposed PostgreSQL Schema

```sql
-- PHASE II ONLY: This table does not exist in Phase I

CREATE TABLE rfl_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    slice VARCHAR(20) NOT NULL,           -- 'U2_env_A', 'U2_env_B', etc.
    policy_version VARCHAR(20),           -- 'v1', 'v2', etc.
    metric_name VARCHAR(50) NOT NULL,     -- e.g., 'p_base', 'p_rfl', 'uplift_delta'
    value DOUBLE PRECISION NOT NULL,
    window_size INTEGER,                  -- NULL for raw, 100 for rolling
    run_id VARCHAR(64),                   -- Links to RFL experiment run
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rfl_metrics_slice_ts ON rfl_metrics (slice, timestamp);
CREATE INDEX idx_rfl_metrics_metric ON rfl_metrics (metric_name, timestamp);
```

---

## 15. PHASE II — Governance & Preregistration

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║              NOT IMPLEMENTED — FUTURE PHASE II                                ║
║                                                                               ║
║  All U2 experiments must adhere to preregistration requirements defined in    ║
║  `PREREG_UPLIFT_U2.yaml`. Metrics cannot be modified after preregistration.   ║
║                                                                               ║
║  ⚠  NO GOVERNANCE INFRASTRUCTURE EXISTS IN PHASE I                            ║
║                                                                               ║
║              PHASE II — NOT RUN IN PHASE I                                    ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### 15.1 Preregistration Fields

| Field | Description |
|-------|-------------|
| `hypothesis` | Null hypothesis and alternative |
| `primary_metric` | `uplift_delta` (locked after registration) |
| `success_threshold` | Per-slice target values |
| `sample_size` | Required cycles for statistical power |
| `alpha` | Significance level (default: 0.05) |
| `stopping_rule` | Early termination criteria (if any) |

### 15.2 Audit Trail Requirements

1. **Preregistration hash:** SHA-256 of `PREREG_UPLIFT_U2.yaml` stored in `manifest.json`
2. **Metric provenance:** Each uplift record links to source JSONL files
3. **Immutable labels:** Slice definitions cannot change mid-experiment
4. **Timestamp integrity:** All records use monotonic UTC timestamps

### 15.3 Manifest Handling

The experiment manifest links preregistration to results:

```json
{
  "manifest_version": "1.0",
  "preregistration": {
    "file": "PREREG_UPLIFT_U2.yaml",
    "hash_sha256": "abc123...",
    "sealed_at": "2025-12-01T00:00:00Z"
  },
  "runs": [
    {
      "slice": "U2_env_A",
      "baseline_run_id": "...",
      "rfl_run_id": "...",
      "completed_at": "2025-12-01T12:00:00Z",
      "result_files": [
        "results/u2_metrics/U2_env_A/baseline_xxx.jsonl",
        "results/u2_metrics/U2_env_A/rfl_yyy.jsonl"
      ]
    }
  ],
  "summary": {
    "generated_at": "2025-12-01T12:30:00Z",
    "slices_completed": 4,
    "slices_significant": 3
  }
}
```

---

*End of Document*
