# Governance History Visualization Dashboard & Storage

**Objective:** Design a dashboard and data storage schema to visualize the state of system governance over time. This will allow reviewers to quickly assess stability and diagnose anomalies based on historical data from sources like `replay_governance_snapshot.json` and other telemetry.

---

## 1. Conceptual Dashboard Design

The dashboard is designed for at-a-glance analysis of system stability. The core principle is to present a multi-layered, time-series view of governance health.

### Key Components:

1.  **Master Control Panel:**
    *   **Time Range Selector:** A dropdown or slider to select the analysis window (e.g., "Last 24 hours," "Last 100 runs," "Custom Range"). This directly addresses the "last N runs" query.
    *   **Layer Filter:** Checkboxes or a multi-select dropdown to toggle the visibility of different system layers (e.g., `P3`, `P4`, `Substrate`).

2.  **System Stability Timeline:**
    *   A compact, horizontal timeline that provides an immediate answer to "Was the system stable?".
    *   Each layer gets a dedicated row.
    *   Time is on the x-axis.
    *   The row is filled with colored blocks corresponding to the `governance_status` for that run:
        *   **Green:** `STABLE`
        *   **Yellow:** `WARNING`
        *   **Red:** `CRITICAL`
    *   Hovering over a block would show a tooltip with the `run_id` and key metrics.

3.  **Critical Metrics Time-Series Plots:**
    *   A grid of charts, one for each key metric (Δp, RSI, Ω, divergence, etc.).
    *   **X-Axis:** Time / Run ID.
    *   **Y-Axis:** Metric value.
    *   Each selected layer is plotted as a differently colored line on the chart, allowing for direct comparison.
    *   Alert thresholds (e.g., a dotted red line for a critical divergence value) can be overlaid on the plots for context.

---

## 2. Governance History Log Schema

A simple, scalable JSONL format is proposed for storing the time-series governance data. Each line is a self-contained JSON object representing a snapshot for a specific layer at a specific time.

### Schema Definition:

*   **`timestamp`** (string): ISO 8601 formatted timestamp of the snapshot.
*   **`run_id`** (string/integer): A unique identifier for the specific governance run or cycle.
*   **`layer`** (string): The system layer being reported (e.g., "P3", "P4", "Substrate").
*   **`governance_status`** (string): A categorical status light ("STABLE", "WARNING", "CRITICAL").
*   **`metrics`** (object): A key-value map of all relevant metrics for that snapshot. This structure is flexible and can accommodate new metrics over time.

### Example Records:

```json
{"timestamp": "2025-12-10T10:00:00Z", "run_id": "run-a4b1c8", "layer": "P4", "governance_status": "STABLE", "metrics": {"delta_p": 0.05, "rsi": 65.2, "omega": 0.98, "divergence": 0.01, "quarantine_ratio": 0.0, "budget_invalid_percent": 0.1}}
```
```json
{"timestamp": "2025-12-10T10:05:00Z", "run_id": "run-a4b1f2", "layer": "P4", "governance_status": "WARNING", "metrics": {"delta_p": 0.15, "rsi": 78.9, "omega": 0.96, "divergence": 0.08, "quarantine_ratio": 0.05, "budget_invalid_percent": 0.8}}
```
