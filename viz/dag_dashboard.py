# viz/dag_dashboard.py
"""
PHASE II - Structural Analytics Dashboard Generator.

This script generates a standalone HTML dashboard to visualize the evolution
metrics and detected anomalies from a Global DAG Builder run.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

# Announce compliance on import
print("PHASE II — NOT USED IN PHASE I: Loading Analytics Dashboard Generator.", file=__import__("sys").stderr)

# Define color scheme for anomalies
ANOMALY_COLORS = {
    "ProofChainCollapse": "red",
    "DepthStagnation": "orange",
    "ExplosiveBranching": "purple",
    "DuplicateProofPattern": "brown",
}

def generate_dashboard(metrics_path: Path, anomalies_path: Path, output_path: Path):
    """
    Generates the HTML dashboard from metrics and anomaly data.
    
    NOTE: Requires `plotly` and `pandas`. Install with:
          `pip install plotly pandas`
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Dashboard generation requires `plotly` and `pandas`. Please run `pip install plotly pandas`.")

    log_info(f"Loading metrics from {metrics_path}...")
    df_metrics = pd.read_json(metrics_path)
    
    log_info(f"Loading anomalies from {anomalies_path}...")
    with open(anomalies_path, "r") as f:
        anomalies = json.load(f)
    df_anomalies = pd.DataFrame(anomalies)

    # --- Create Figures ---
    fig_growth = plot_node_edge_growth(df_metrics, df_anomalies)
    fig_deltas = plot_deltas(df_metrics)
    fig_depth = plot_max_depth(df_metrics, df_anomalies)
    fig_branching = plot_branching_factor(df_metrics, df_anomalies)

    # --- Assemble HTML ---
    log_info(f"Assembling HTML dashboard to {output_path}...")
    with open(output_path, "w") as f:
        f.write("<html><head><title>Structural Analytics Dashboard</title></head>")
        f.write("<body style='font-family: sans-serif; background-color: #f4f4f4;'>")
        f.write("<h1>Structural Analytics Dashboard</h1>")
        
        # Convert figures to HTML and write them
        f.write(fig_growth.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_depth.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_deltas.to_html(full_html=False, include_plotlyjs='cdn'))
        f.write(fig_branching.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write("</body></html>")
    log_info("Dashboard generation complete.")

def add_anomaly_markers(fig, df_anomalies):
    """Helper to add anomaly markers to a figure."""
    if df_anomalies.empty:
        return
        
    for anomaly_type, color in ANOMALY_COLORS.items():
        anomalies_of_type = df_anomalies[df_anomalies['anomaly_type'] == anomaly_type]
        if not anomalies_of_type.empty:
            fig.add_trace(go.Scatter(
                x=anomalies_of_type['cycle'],
                y=anomalies_of_type['cycle'].apply(lambda x: 0), # Position at bottom
                mode='markers',
                marker_symbol='x',
                marker_color=color,
                marker_size=10,
                name=anomaly_type,
                hoverinfo='text',
                text=[f"{row['anomaly_type']} at cycle {row['cycle']}" for _, row in anomalies_of_type.iterrows()]
            ))

def plot_node_edge_growth(df, df_anomalies):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df['cycle'], y=df['Nodes(t)'], name="Total Nodes", mode='lines'), secondary_y=False)
    fig.add_trace(go.Scatter(x=df['cycle'], y=df['Edges(t)'], name="Total Edges", mode='lines'), secondary_y=True)
    add_anomaly_markers(fig, df_anomalies)
    fig.update_layout(title_text="Node and Edge Growth")
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="Total Nodes", secondary_y=False)
    fig.update_yaxes(title_text="Total Edges", secondary_y=True)
    return fig

def plot_deltas(df):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['cycle'], y=df['ΔNodes(t)'], name='New Nodes (ΔNodes)'))
    fig.add_trace(go.Bar(x=df['cycle'], y=df['ΔEdges(t)'], name='New Edges (ΔEdges)'))
    fig.update_layout(title_text="Discovery Rate per Cycle", barmode='group')
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="Count")
    return fig

def plot_max_depth(df, df_anomalies):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['cycle'], y=df['MaxDepth(t)'], name="Max Proof Depth", mode='lines+markers'))
    add_anomaly_markers(fig, df_anomalies)
    fig.update_layout(title_text="Maximum Proof Depth Evolution")
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="Depth")
    return fig

def plot_branching_factor(df, df_anomalies):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['cycle'], y=df['GlobalBranchingFactor(t)'], name="Global Branching Factor"))
    add_anomaly_markers(fig, df_anomalies)
    fig.update_layout(title_text="Global Branching Factor")
    fig.update_xaxes(title_text="Cycle")
    fig.update_yaxes(title_text="Ratio (Edges/Nodes)")
    return fig

def log_info(msg):
    print(f"[INFO] {msg}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Generate a structural analytics dashboard from DAG metrics.")
    parser.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="Path to the global_dag_metrics.json file."
    )
    parser.add_argument(
        "--anomalies",
        type=Path,
        required=True,
        help="Path to the anomaly_report.json file."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default="dashboard.html",
        help="Path to write the output HTML dashboard file."
    )
    args = parser.parse_args()

    generate_dashboard(args.metrics, args.anomalies, args.out)

if __name__ == "__main__":
    main()
