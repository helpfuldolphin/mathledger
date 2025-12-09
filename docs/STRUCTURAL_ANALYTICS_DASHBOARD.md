# STRUCTURAL_ANALYTICS_DASHBOARD.md

**Version:** 1.0
**Status:** DRAFT
**Owner:** Gemini G (Global DAG Architect)

## 1. Overview

This document specifies the design and implementation of the **Structural Analytics Dashboard**. This dashboard serves as the primary interface for visualizing the structural evolution and health of the Global Proof Dependency Graph (`DAG_global`) during a U2 experiment run.

The dashboard will consume the `global_dag_metrics.json` and `anomaly_report.json` artifacts produced by the `run_uplift_u2.py` runner and its integrated `GlobalDagBuilder` and `AnomalyDetector` components. Its purpose is to provide immediate, actionable insights into the behavior of the RFL discovery engine.

## 2. Dashboard Components

The dashboard will be generated as a standalone HTML file containing interactive plots and tables. The implementation will be a Python script (`viz/dag_dashboard.py`) using the `plotly` library for its rich interactivity and ability to export to HTML.

### 2.1. Header Information

The dashboard will begin with a summary header containing key metadata from the experiment run:

- **Experiment ID:** (e.g., `uplift_u2_tree_001`)
- **Slice Name:** (e.g., `slice_uplift_tree`)
- **Run Type:** (e.g., `paired`, `baseline`)
- **Total Cycles:** (e.g., `500`)
- **Timestamp (UTC):**
- **Total Anomalies Detected:**

### 2.2. Primary Time-Series Plots

This section will feature a series of interactive time-series plots, with `cycle` as the shared x-axis. Each plot will visualize one of the core evolution metrics from `global_dag_metrics.json`.

1.  **Node and Edge Growth:**
    *   **Content:** A dual-axis plot showing `Nodes(t)` and `Edges(t)` over time.
    *   **Y1-Axis:** Total Nodes (count)
    *   **Y2-Axis:** Total Edges (count)
    *   **Purpose:** Visualizes the overall growth rate of the knowledge graph.

2.  **Discovery Rate (Deltas):**
    *   **Content:** A bar chart showing `ΔNodes(t)` and `ΔEdges(t)` per cycle.
    *   **Y-Axis:** New Items (count)
    *   **Purpose:** Highlights cycles with high rates of new discovery versus cycles of consolidation or redundant work.

3.  **Maximum Proof Depth Evolution:**
    *   **Content:** A line chart showing `MaxDepth(t)` over time.
    *   **Y-Axis:** Depth (count)
    *   **Purpose:** Tracks the RFL engine's ability to find deeper, more complex lines of reasoning. Stagnation or collapse is immediately visible.

4.  **Global Branching Factor:**
    *   **Content:** A line chart showing `GlobalBranchingFactor(t)`.
    *   **Y-Axis:** Ratio (float)
    *   **Purpose:** Monitors the average complexity of derivations. A rising factor may indicate more complex proofs, while a falling factor may indicate a focus on simpler, linear chains.

### 2.3. Anomaly Timeline

This component is critical for correlating structural events with the evolution metrics.

*   **Content:** An event plot or a series of vertical lines/markers overlaid on the primary time-series plots (especially the `MaxDepth(t)` plot).
*   **Data Source:** `anomaly_report.json`.
*   **Logic:**
    *   For each anomaly in the report, a marker will be placed on the x-axis at the corresponding `cycle`.
    *   Hovering over the marker will display a tooltip with the `anomaly_type` and its details (e.g., `"ProofChainCollapse", "delta_max_depth": -8`).
    *   Different anomaly types will have distinct colors and symbols for easy identification.

### 2.4. Slice-Level Comparative View (for Paired Runs)

When analyzing a `paired` experiment, the dashboard will include comparative plots to directly assess the performance of `RFL` vs. `baseline`.

*   **Content:** A series of plots, each showing the `baseline` and `rfl` metric on the same axes.
    *   Plot 1: `MaxDepth(t)` for baseline vs. RFL.
    *   Plot 2: `Nodes(t)` for baseline vs. RFL.
    *   Plot 3: `ΔNodes(t)` (as a bar chart) for baseline vs. RFL.
*   **Purpose:** Provides a direct, visual answer to the core experimental question: Is the RFL policy outperforming the baseline in structural exploration?

## 3. Skeleton Implementation Plan (`viz/dag_dashboard.py`)

The skeleton script will provide a proof-of-concept for generating the dashboard.

1.  **Dependencies:**
    *   `plotly`: For creating interactive plots.
    *   `pandas`: For easy data manipulation of the metrics JSON.

2.  **Script Structure:**
    *   A `main` function driven by `argparse` to accept paths to the `metrics.json` and `anomalies.json` files.
    *   A `load_data` function to read the JSON files into pandas DataFrames.
    *   A function for each plot type (e.g., `plot_node_growth(df)`, `plot_max_depth(df, anomalies)`).
    *   Each plotting function will:
        *   Create a `plotly.graph_objects.Figure`.
        *   Add traces (lines, bars) for the metrics.
        *   Add scatter traces or shapes for the anomaly markers.
        *   Configure layout, titles, and axes.
        *   Return the figure object.
    *   The `main` function will orchestrate the creation of all figures and combine them into a single HTML output file using `fig.to_html(full_html=False, include_plotlyjs='cdn')` and assembling the final HTML structure.

## 4. Example Visualization Snippet (Conceptual)

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add Node Growth trace
fig.add_trace(
    go.Scatter(x=df['cycle'], y=df['Nodes(t)'], name="Total Nodes"),
    secondary_y=False,
)

# Add Edge Growth trace
fig.add_trace(
    go.Scatter(x=df['cycle'], y=df['Edges(t)'], name="Total Edges"),
    secondary_y=True,
)

# Add anomaly markers
anomalies_df = df_anomalies[df_anomalies['anomaly_type'] == 'ExplosiveBranching']
fig.add_trace(
    go.Scatter(
        x=anomalies_df['cycle'], 
        y=[0]*len(anomalies_df), # Position at bottom
        mode='markers',
        marker_symbol='x',
        marker_color='red',
        name='Explosive Branching'
    ),
    secondary_y=False,
)

# Set titles and layout...
fig.show()
```
This specification provides a clear blueprint for constructing a valuable diagnostic and analytical tool, directly translating the raw structural telemetry into interpretable scientific insights.
