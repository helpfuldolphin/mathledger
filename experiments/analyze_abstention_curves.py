#!/usr/bin/env python3
"""
Abstention Dynamics Analyst (GEMINI-D) Tool.

Analyzes JSONL logs from First-Order (FO) cycle runs to compare abstention behavior
between Baseline (no RFL) and RFL-enabled runs.

Metrics:
    - Abstention Rate A(t): Rolling probability of abstention.
    - Cumulative Abstentions C(t): Total abstentions over time.
    - Method Distribution: Usage of different verification methods.

Hypothesis:
    RFL reduces abstention rate A(t) relative to Baseline after a burn-in period.

Usage:
    python experiments/analyze_abstention_curves.py \
        --baseline results/fo_baseline.jsonl \
        --rfl results/fo_rfl.jsonl \
        --window-size 100 \
        --burn-in 200
"""

import json
import argparse
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Configure default plot style for paper-grade figures
plt.style.use('fast')  # Clean, minimal style
ARTIFACTS_DIR = Path("artifacts/figures")

def is_abstention(row: pd.Series) -> bool:
    """
    Determines if a log entry represents an abstention.
    
    Logic:
    1. Explicit 'status' == 'abstain' (case insensitive)
    2. Fallback: 'method' (or 'verification_method') == 'lean-disabled'
    """
    # 1. Normalize fields
    status = str(row.get("status", "")).lower()
    method = row.get("method")
    if pd.isna(method) or method == "":
        method = row.get("verification_method", "")
    if pd.isna(method):
        method = ""
    
    # 2. Predicate
    if status == "abstain":
        return True
    
    return method == "lean-disabled"

def normalize_method(row: pd.Series) -> str:
    """Extracts a consistent method name for distribution analysis."""
    m = row.get("method")
    if pd.isna(m) or m == "":
        m = row.get("verification_method", "unknown")
    return str(m)

def load_logs(filepath: str) -> pd.DataFrame:
    """Reads a JSONL file and returns a normalized DataFrame."""
    data = []
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON at line {i+1} in {filepath}", file=sys.stderr)
                continue
    
    if not data:
        print(f"Error: No valid data found in {filepath}", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(data)
    
    # Ensure 'cycle' exists
    if 'cycle' not in df.columns:
        # If 'cycle' is missing, assume sequential index
        print(f"Note: 'cycle' column missing in {filepath}, using index.", file=sys.stderr)
        df['cycle'] = df.index
        
    # Apply abstention predicate
    df['is_abstention'] = df.apply(is_abstention, axis=1).astype(int)
    
    # Normalize method column for summary
    df['normalized_method'] = df.apply(normalize_method, axis=1)

    return df.sort_values('cycle')

def compute_metrics(df: pd.DataFrame, label: str, window_size: int) -> pd.DataFrame:
    """Computes rolling averages and cumulative sums."""
    df = df.copy()
    df['run_type'] = label
    
    # Rolling Mean Abstention Rate
    # min_periods=1 ensures we get data from the start, though it's noisy initially
    df['abstention_rate_rolling'] = df['is_abstention'].rolling(window=window_size, min_periods=1).mean()
    
    # Cumulative Abstentions
    df['cumulative_abstentions'] = df['is_abstention'].cumsum()
    
    return df

def plot_abstention_rate(baseline_df: pd.DataFrame, rfl_df: pd.DataFrame, window_size: int, burn_in: int):
    """Generates Figure 1: Abstention Rate vs Cycle."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(baseline_df['cycle'], baseline_df['abstention_rate_rolling'], 
             label='Baseline (No RFL)', color='blue', linestyle='-', alpha=0.8, linewidth=1.5)
    plt.plot(rfl_df['cycle'], rfl_df['abstention_rate_rolling'], 
             label='RFL Enabled', color='red', linestyle='-', alpha=0.8, linewidth=1.5)
    
    plt.title(f'RFL vs Baseline: Rolling Abstention Rate (W={window_size}, burn-in={burn_in})')
    plt.xlabel('Cycle Index')
    plt.ylabel(f'Abstention Rate (P(abstain))')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outfile = ARTIFACTS_DIR / "rfl_abstention_rate.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved figure: {outfile}")
    plt.close()

def plot_cumulative_abstentions(baseline_df: pd.DataFrame, rfl_df: pd.DataFrame):
    """Generates Figure 2: Cumulative Abstentions vs Cycle."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(baseline_df['cycle'], baseline_df['cumulative_abstentions'], 
             label='Baseline (No RFL)', color='blue', linestyle='--')
    plt.plot(rfl_df['cycle'], rfl_df['cumulative_abstentions'], 
             label='RFL Enabled', color='red', linestyle='-')
    
    plt.title('Cumulative Abstentions Growth (Baseline vs RFL)')
    plt.xlabel('Cycle Index')
    plt.ylabel('Total Abstentions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    outfile = ARTIFACTS_DIR / "rfl_cumulative_abstentions.png"
    plt.savefig(outfile, dpi=150)
    print(f"Saved figure: {outfile}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze abstention dynamics from FO logs.")
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline JSONL logs')
    parser.add_argument('--rfl', type=str, required=True, help='Path to RFL JSONL logs')
    parser.add_argument('--window-size', type=int, default=100, help='Rolling window size for abstention rate')
    parser.add_argument('--burn-in', type=int, default=200, help='Cycles to ignore for steady-state comparison')
    parser.add_argument('--out-summary', type=str, help='Path to write summary JSON/CSV')
    
    args = parser.parse_args()

    # Ensure output directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading logs...\n  Baseline: {args.baseline}\n  RFL:      {args.rfl}")
    df_baseline = load_logs(args.baseline)
    df_rfl = load_logs(args.rfl)

    print("Computing metrics...")
    df_baseline = compute_metrics(df_baseline, 'Baseline', args.window_size)
    df_rfl = compute_metrics(df_rfl, 'RFL', args.window_size)

    # --- Figures ---
    print("Generating figures...")
    plot_abstention_rate(df_baseline, df_rfl, args.window_size, args.burn_in)
    plot_cumulative_abstentions(df_baseline, df_rfl)

    # --- Statistical Summary ---
    
    def get_recent_stats(df, burn_in):
        # Filter for cycles > burn_in
        subset = df[df['cycle'] > burn_in]
        if subset.empty:
            return None, 0
        return subset['is_abstention'].mean(), len(subset)

    base_rate, base_n = get_recent_stats(df_baseline, args.burn_in)
    rfl_rate, rfl_n = get_recent_stats(df_rfl, args.burn_in)
    
    delta = 0.0
    if base_n > 0 and rfl_n > 0:
        delta = rfl_rate - base_rate

    # Micro-summary for the paper
    print(f"\n---- Abstention Summary (last {rfl_n} cycles after burn-in={args.burn_in}) ----")
    if base_n == 0 or rfl_n == 0:
        print(f"WARNING: Insufficient data post-burn-in (> {args.burn_in} cycles).")
    else:
        print(f"Baseline: Ā = {base_rate:.3f}")
        print(f"RFL:      Ā = {rfl_rate:.3f}")
        print(f"ΔA = {delta:+.3f}")

    # Method Distribution
    print("\n---- Method Distribution (Overall) ----")
    for label, df in [('Baseline', df_baseline), ('RFL', df_rfl)]:
        dist = df['normalized_method'].value_counts(normalize=True)
        print(f"\n{label} Methods:")
        for method, freq in dist.items():
            print(f"  {method:<20}: {freq:.1%}")

    # --- Optional Output File ---
    if args.out_summary:
        summary_data = {
            "metric": "abstention_rate_post_burnin",
            "burn_in_cycles": args.burn_in,
            "baseline_rate": base_rate,
            "rfl_rate": rfl_rate,
            "delta": delta,
            "baseline_n": base_n,
            "rfl_n": rfl_n,
            "baseline_methods": df_baseline['normalized_method'].value_counts(normalize=True).to_dict(),
            "rfl_methods": df_rfl['normalized_method'].value_counts(normalize=True).to_dict()
        }
        
        with open(args.out_summary, 'w') as f:
            if args.out_summary.endswith('.json'):
                json.dump(summary_data, f, indent=2)
            else:
                # Simple CSV format for the main metrics
                f.write("metric,value\n")
                f.write(f"baseline_rate,{base_rate}\n")
                f.write(f"rfl_rate,{rfl_rate}\n")
                f.write(f"delta,{delta}\n")
        print(f"\nSummary written to {args.out_summary}")

if __name__ == "__main__":
    main()