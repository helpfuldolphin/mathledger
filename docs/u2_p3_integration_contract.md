# U2 → P3 Integration Contract

**Version**: 1.0.0  
**Author**: Manus-F, Deterministic Planner & Substrate Executor Engineer  
**Date**: 2025-12-06  
**Status**: Final

---

## 1. Overview

This document specifies the integration contract for deriving P3-level performance metrics (Δp, RSI, Ω) from the raw JSONL trace logs produced by the U2 Planner. It defines the required data fields, the precise algorithms for metric calculation, and the deterministic ordering constraints necessary to guarantee reproducible analysis.

Adherence to this contract ensures that any P3-compliant analysis tool can consume U2 trace logs and produce identical, verifiable metrics, regardless of the analysis environment.

## 2. Trace Log Format Contract

The U2 trace log is a JSONL file where each line is a JSON object representing an event. For P3 metric derivation, only events with `"event_type": "execution"` are required. Each `execution` event object **MUST** contain the following fields and data types:

| Field Path | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `event_type` | String | Must be `"execution"`. | `"execution"` |
| `cycle` | Integer | The execution cycle number. | `10` |
| `timestamp_ms` | Integer | Millisecond timestamp of the event. | `1670356800000` |
| `worker_id` | Integer | The ID of the worker that performed the execution. | `3` |
| `data.outcome` | String | The result of the execution. Must be one of `"success"`, `"failure"`, `"error"`. | `"success"` |
| `data.is_tautology` | Boolean | `true` if the statement was proven to be a tautology. | `true` |
| `data.time_ms` | Integer | The wall-clock time in milliseconds for the execution step. | `15` |
| `data.statement.hash` | String | The canonical SHA-256 hash of the statement that was executed. | `"a1b2..."` |
| `data.statement.normalized` | String | The canonically-formatted string representation of the statement. | `"p→(q→p)"` |

### Example `execution` Event:

```json
{"event_type": "execution", "cycle": 10, "timestamp_ms": 1670356800123, "worker_id": 3, "data": {"outcome": "success", "is_tautology": true, "time_ms": 15, "statement": {"hash": "a1b2c3d4...", "normalized": "p→(q→p)", ...}, ...}}
```

## 3. Deterministic Processing Constraints

To ensure reproducible metric calculation, trace files **MUST** be processed in a deterministic order. The canonical processing order is achieved by sorting the events (lines) of the JSONL file based on the following criteria, in ascending order:

1.  **`cycle`** (Integer)
2.  **`worker_id`** (Integer)
3.  **`data.statement.hash`** (String, lexicographical)

This sorting ensures that events are grouped by cycle and worker, and that within each worker-cycle, the processing order is stable and independent of non-deterministic factors like `timestamp_ms`.

## 4. P3 Metric Derivation Algorithms

All metrics are derived from the set of `execution` events in the trace log, hereafter referred to as *E*.

### 4.1. Ω (Omega): The Knowledge Base

Ω is defined as the set of all unique statements proven to be tautologies during the experiment.

**Algorithm**:
1.  Initialize an empty set, `proven_hashes = {}`.
2.  Filter *E* to create a subset *S* where `event.data.is_tautology == true`.
3.  For each `event` in *S*:
    a. Add `event.data.statement.hash` to `proven_hashes`.
4.  Ω is the set `proven_hashes`.

**Output**: A set of SHA-256 hashes.

### 4.2. Δp (Delta-p): Change in Proven Statements

Δp is defined as the total number of newly proven, unique statements.

**Algorithm**:
1.  Derive Ω as specified in section 4.1.
2.  Δp is the cardinality (size) of the set Ω.

**Output**: A single integer.

### 4.3. RSI (Reasoning Step Intensity)

RSI is defined as the number of reasoning steps (executions) performed per second.

**Algorithm**:
1.  Calculate `total_executions` as the total number of events in *E*.
2.  Find `min_timestamp` = `min(event.timestamp_ms for event in E)`.
3.  Find `max_timestamp` = `max(event.timestamp_ms for event in E)`.
4.  Calculate `total_wall_time_seconds` = `(max_timestamp - min_timestamp) / 1000.0`.
5.  If `total_wall_time_seconds` is zero, RSI is zero.
6.  Otherwise, `RSI = total_executions / total_wall_time_seconds`.

**Output**: A single floating-point number.

## 5. Contract Summary

| Metric | Definition | Derivation Source | Key Fields | Deterministic? |
| :--- | :--- | :--- | :--- | :--- |
| **Ω** | Set of unique proven statements | `data.statement.hash` where `data.is_tautology == true` | `is_tautology`, `statement.hash` | Yes |
| **Δp** | Count of unique proven statements | `size(Ω)` | `is_tautology`, `statement.hash` | Yes |
| **RSI** | Executions per second | `count(E) / wall_time` | `timestamp_ms` | No* |

*Note: RSI is inherently non-deterministic as it depends on wall-clock time. However, the calculation method is standardized by this contract. The deterministic components of the experiment (Δp, Ω) are the primary metrics for reproducibility.
