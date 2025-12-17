# scripts/generate_dag_health_report.py
"""
PHASE III - Periodic DAG Health Report Generator.

This script scans the posture history archive, computes long-term trends
and aggregate statistics, and generates a structured JSON health report.

It is intended to be run periodically (e.g., weekly) by a scheduled CI job.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import glob

# Add project root for local imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.posture_analysis import build_dag_posture_timeline

HISTORY_DIR = Path("artifacts/dag/posture_history")
REPORTS_DIR = Path("artifacts/reports")

def main():
    """Main function to generate the health report."""
    print("--- Generating DAG Health Report ---", file=sys.stderr)

    # 1. Glob all historical snapshots
    snapshot_files = sorted(glob.glob(str(HISTORY_DIR / "*.json")))
    
    if not snapshot_files:
        print("[WARN] No posture snapshots found in history. Skipping report.", file=sys.stderr)
        return

    print(f"Found {len(snapshot_files)} snapshots in history.", file=sys.stderr)

    # 2. Load snapshots into a list
    snapshots = []
    for file_path in snapshot_files:
        try:
            with open(file_path, 'r') as f:
                snapshots.append(json.load(f))
        except (json.JSONDecodeError, IOError) as e:
            print(f"[WARN] Could not load or parse snapshot '{file_path}': {e}", file=sys.stderr)
            continue

    # 3. Call the timeline builder
    timeline_analysis = build_dag_posture_timeline(snapshots)

    # 4. Format the final health report
    start_ts = timeline_analysis["timeline"][0]["posture"].get("timestamp") if timeline_analysis["timeline"] else None
    end_ts = timeline_analysis["timeline"][-1]["posture"].get("timestamp") if timeline_analysis["timeline"] else None
    
    # Calculate overall net changes
    first_posture = timeline_analysis["timeline"][0]["posture"]
    last_posture = timeline_analysis["timeline"][-1]["posture"]
    net_depth_change = last_posture.get("max_depth", 0) - first_posture.get("max_depth", 0)
    net_vertex_growth = last_posture.get("vertex_count", 0) - first_posture.get("vertex_count", 0)

    aggregates = timeline_analysis["aggregates"]
    eligibility_rate = aggregates["eligible_count"] / aggregates["total_snapshots"] if aggregates["total_snapshots"] > 0 else 0

    report = {
        "report_metadata": {
            "report_id": f"weekly_{datetime.now(timezone.utc).strftime('%Y_%U')}",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
        },
        "aggregate_summary": {
            "total_snapshots_analyzed": aggregates["total_snapshots"],
            "eligibility_rate": round(eligibility_rate, 4),
            "net_depth_change": net_depth_change,
            "net_vertex_growth": net_vertex_growth,
            "periods_with_depth_growth": aggregates["positive_depth_delta_periods"],
            "periods_with_depth_regression": aggregates["negative_depth_delta_periods"],
        },
        "detected_trends": timeline_analysis["trend_flags"],
        # Include a sample of timeline points for quick view, not all of them
        "timeline_sample": [
            {"timestamp": entry["posture"].get("timestamp"), "max_depth": entry["posture"].get("max_depth")}
            for entry in timeline_analysis["timeline"]
        ]
    }

    # 5. Save the report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_filename = f"dag_health_report_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.json"
    output_path = REPORTS_DIR / report_filename

    print(f"Saving health report to: {output_path}", file=sys.stderr)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
        
    print("\n--- DAG Health Report Generation Complete ---", file=sys.stderr)

if __name__ == "__main__":
    main()
