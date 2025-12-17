#!/usr/bin/env python3
"""
Budget Drift vs Divergence Cross-Plot Generator (P5 Calibration).

Generates a cross-plot dataset comparing budget stability_index against divergence_rate
across CAL-EXP calibration experiment windows. This enables visualization of the
correlation between budget invariant health ("Energy Law") and twin-vs-real divergence.

Inputs:
  - CAL-EXP-* reports (P3, P4) with budget_storyline_summary and windows data

Output:
  - JSON or CSV with columns: window_idx, stability_index, divergence_rate

SHADOW MODE CONTRACT:
- This script is observational only
- Outputs are for analysis and visualization
- No gating decisions based on this data in Phase X/P5 POC
- Phase Y only for P5AcceptanceGate integration

Usage:
    python scripts/plot_budget_vs_divergence.py \
        --cal-exp1 cal_exp1_report.json \
        --cal-exp2 cal_exp2_report.json \
        --output budget_vs_divergence.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from derivation.budget_cal_exp_integration import (
        extract_budget_storyline_from_cal_exp_report,
    )
except ImportError as e:
    print(f"ERROR: Failed to import budget cal-exp integration: {e}", file=sys.stderr)
    sys.exit(1)


def load_cal_exp_report(path: Path) -> Dict[str, Any]:
    """Load CAL-EXP report from JSON file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Report file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in report: {e}", file=sys.stderr)
        sys.exit(1)


def extract_cross_plot_data(
    cal_exp_reports: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Extract cross-plot data from CAL-EXP reports.
    
    Extracts (window_idx, stability_index, divergence_rate) tuples from
    calibration experiment reports that include budget_storyline_summary.
    
    Args:
        cal_exp_reports: List of CAL-EXP report dicts.
    
    Returns:
        List of dicts with keys: window_idx, stability_index, divergence_rate,
        budget_combined_status, budget_projection_class, budget_confounded,
        budget_confound_reason, experiment_id, run_id
        
    Note:
        Windows are sorted by (experiment_id, window_idx) for deterministic ordering.
    """
    cross_plot_data: List[Dict[str, Any]] = []
    
    for report in cal_exp_reports:
        # Extract budget storyline summary
        budget_storyline = extract_budget_storyline_from_cal_exp_report(report)
        
        if budget_storyline is None:
            # Skip reports without budget storyline
            continue
        
        experiment_id = budget_storyline.get("experiment_id", "UNKNOWN")
        run_id = budget_storyline.get("run_id", "UNKNOWN")
        stability_index = budget_storyline.get("stability_index", 0.0)
        
        # Extract budget fields for all windows
        combined_status = budget_storyline.get("combined_status", "OK")
        projection_class = budget_storyline.get("projection_class", "UNKNOWN")
        
        # Extract windows data (use annotated windows if available)
        windows = report.get("windows", [])
        
        # Map each window to a data point
        # Use window_idx from window dict if available (for determinism), otherwise enumerate
        for idx, window in enumerate(windows):
            # Prefer explicit window index field if present, otherwise use enumerate index
            window_idx = window.get("window_idx", idx)
            divergence_rate = window.get("divergence_rate", 0.0)
            
            # Use window-level budget fields if annotated, otherwise fall back to summary-level
            window_combined_status = window.get("budget_combined_status", combined_status)
            window_stability_index = window.get("budget_stability_index", stability_index)
            window_projection_class = window.get("budget_projection_class", projection_class)
            window_confounded = window.get("budget_confounded", False)  # Default to False if not annotated
            window_confound_reason = window.get("budget_confound_reason", None)
            
            cross_plot_data.append({
                "window_idx": window_idx,
                "stability_index": window_stability_index,
                "divergence_rate": divergence_rate,
                "budget_combined_status": window_combined_status,
                "budget_projection_class": window_projection_class,
                "budget_confounded": window_confounded,
                "budget_confound_reason": window_confound_reason,
                "experiment_id": experiment_id,
                "run_id": run_id,
            })
    
    return cross_plot_data


def write_csv_output(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write cross-plot data to CSV file."""
    if not data:
        print("WARNING: No data to write", file=sys.stderr)
        return
    
    fieldnames = [
        "window_idx",
        "stability_index",
        "divergence_rate",
        "budget_combined_status",
        "budget_projection_class",
        "budget_confounded",
        "budget_confound_reason",
        "experiment_id",
        "run_id",
    ]
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def write_json_output(data: List[Dict[str, Any]], output_path: Path) -> None:
    """Write cross-plot data to JSON file."""
    output = {
        "schema_version": "1.0.0",
        "data": data,
        "summary": {
            "total_points": len(data),
            "experiments": list(set(d["experiment_id"] for d in data)),
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Budget Drift vs Divergence Cross-Plot Generator (P5 Calibration)"
    )
    parser.add_argument(
        "--cal-exp1",
        type=Path,
        help="Path to CAL-EXP-1 report JSON",
    )
    parser.add_argument(
        "--cal-exp2",
        type=Path,
        help="Path to CAL-EXP-2 report JSON (optional)",
    )
    parser.add_argument(
        "--cal-exp3",
        type=Path,
        help="Path to CAL-EXP-3 report JSON (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path (CSV or JSON, determined by extension)",
    )
    
    args = parser.parse_args()
    
    # Collect reports
    reports: List[Dict[str, Any]] = []
    
    if args.cal_exp1:
        reports.append(load_cal_exp_report(args.cal_exp1))
    
    if args.cal_exp2:
        reports.append(load_cal_exp_report(args.cal_exp2))
    
    if args.cal_exp3:
        reports.append(load_cal_exp_report(args.cal_exp3))
    
    if not reports:
        print("ERROR: At least one CAL-EXP report must be provided", file=sys.stderr)
        return 1
    
    # Extract cross-plot data
    cross_plot_data = extract_cross_plot_data(reports)
    
    if not cross_plot_data:
        print("WARNING: No cross-plot data extracted (reports may lack budget_storyline_summary)", file=sys.stderr)
        return 0
    
    # Sort by window_idx for deterministic ordering
    cross_plot_data.sort(key=lambda x: (x.get("experiment_id", ""), x.get("window_idx", 0)))
    
    # Write output
    output_path = Path(args.output)
    if output_path.suffix.lower() == ".csv":
        write_csv_output(cross_plot_data, output_path)
    else:
        write_json_output(cross_plot_data, output_path)
    
    print(f"Cross-plot data written to: {output_path}")
    print(f"Total data points: {len(cross_plot_data)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



