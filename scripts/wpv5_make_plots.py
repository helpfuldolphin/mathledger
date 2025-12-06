#!/usr/bin/env python3
"""
Robust plotter for WPV5 throughput analysis.
Uses only proofs_per_hour and Run # on X-axis; no depth analysis.
Never computes NaN/Inf limits - handles edge cases gracefully.
"""
import os, csv, sys
import matplotlib.pyplot as plt
import numpy as np

BL = "artifacts/wpv5/baseline_runs.csv"
GD = "artifacts/wpv5/guided_runs.csv"
OUT = "artifacts/wpv5/throughput_vs_depth.png"

def safe_float(value, default=0.0):
    """Safely convert value to float, handling None, empty strings, and invalid values."""
    if value is None or value == "":
        return default
    try:
        result = float(value)
        # Check for NaN or Inf
        if np.isnan(result) or np.isinf(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def read_pph(path):
    """Read proofs_per_hour values from CSV, handling all edge cases."""
    rows = []
    if not os.path.exists(path):
        return rows

    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pph = safe_float(row.get("proofs_per_hour"))
                if pph > 0:  # Only include positive values
                    rows.append(pph)
    except Exception as e:
        print(f"Warning: Error reading {path}: {e}", file=sys.stderr)

    return rows

def create_plot(baseline_data, guided_data, output_path):
    """Create a robust plot that never computes NaN/Inf limits."""
    plt.figure(figsize=(8, 5))

    # Plot baseline data if available
    if baseline_data:
        x_baseline = list(range(1, len(baseline_data) + 1))
        plt.plot(x_baseline, baseline_data, marker="o", label="Baseline",
                linewidth=2, markersize=6, color="blue")

    # Plot guided data if available
    if guided_data:
        x_guided = list(range(1, len(guided_data) + 1))
        plt.plot(x_guided, guided_data, marker="s", label="Guided",
                linewidth=2, markersize=6, color="red")

    # Set labels and formatting
    plt.xlabel("Run #", fontsize=12)
    plt.ylabel("Proofs/hour", fontsize=12)
    plt.title("Throughput Comparison: Baseline vs Guided", fontsize=14, fontweight="bold")

    # Add grid and legend
    plt.grid(True, alpha=0.3)
    if baseline_data or guided_data:
        plt.legend(fontsize=11)

    # Set axis limits safely - never use NaN/Inf
    if baseline_data or guided_data:
        all_values = baseline_data + guided_data
        min_val = min(all_values)
        max_val = max(all_values)

        # Add 10% padding, but ensure we don't create invalid ranges
        if min_val == max_val:
            # Single value case - create a small range
            padding = max(1.0, min_val * 0.1)
            plt.ylim(max(0, min_val - padding), min_val + padding)
        else:
            # Multiple values - add padding
            range_val = max_val - min_val
            padding = range_val * 0.1
            plt.ylim(max(0, min_val - padding), max_val + padding)

    # Set x-axis limits
    max_runs = max(len(baseline_data), len(guided_data))
    if max_runs > 0:
        plt.xlim(0.5, max_runs + 0.5)

    plt.tight_layout()

    # Save with error handling
    try:
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving plot: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        plt.close()

def main():
    """Main function with comprehensive error handling."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Read data
    baseline_data = read_pph(BL)
    guided_data = read_pph(GD)

    # Handle empty data case
    if not baseline_data and not guided_data:
        print("No experiment data found. Creating placeholder plot...")
        plt.figure(figsize=(6, 3.5))
        plt.text(0.5, 0.5, "No data yet", ha="center", va="center", fontsize=16)
        plt.axis("off")
        plt.title("Throughput Analysis - No Data Available", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUT, bbox_inches="tight")
        print(f"Placeholder plot saved to: {OUT}")
        return

    # Create the main plot
    create_plot(baseline_data, guided_data, OUT)

    # Print summary statistics
    if baseline_data:
        print(f"Baseline: {len(baseline_data)} runs, avg {np.mean(baseline_data):.1f} proofs/hour")
    if guided_data:
        print(f"Guided: {len(guided_data)} runs, avg {np.mean(guided_data):.1f} proofs/hour")

if __name__ == "__main__":
    main()
