#!/usr/bin/env python3
"""
Hash State Drift Analyzer
=========================

Analyzes the scaling behavior of Delta H (Hamming distance between H_t and H_{t-1})
as a function of verified volume N_v.

Sanity Check Hypothesis:
    beta approx 0  (Random Avalanche)
    avg(Delta H) approx 128 bits

Usage:
    python scripts/analyze_ht_scaling.py --input reports/ht_dynamics.csv
"""

import argparse
import csv
import math
import sys
import numpy as np

def hamming_distance(hex1: str, hex2: str) -> int:
    """Compute bitwise Hamming distance between two 64-char hex strings (256 bits)."""
    if not hex1 or not hex2:
        return 0
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    xor = val1 ^ val2
    # Python 3.10+ has int.bit_count(), fallback for older versions if needed
    if hasattr(int, "bit_count"):
        return xor.bit_count()
    else:
        return bin(xor).count('1')

def analyze(input_file: str, min_nv: int, min_dh: int):
    sequences = []
    n_values = []
    delta_h_values = []
    
    prev_ht = None
    row_count = 0
    skipped_count = 0

    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_count += 1
            try:
                seq = int(row['sequence'])
                ht = row['ht_hex'].strip()
                nv = int(row['nv'])
            except (ValueError, KeyError):
                skipped_count += 1
                continue

            if len(ht) != 64:
                # Warn once or verbose? Let's just count skippable bad rows
                skipped_count += 1
                continue
            
            if prev_ht is not None:
                # Skip if Nv is too small (early warmup)
                if nv < min_nv:
                    prev_ht = ht
                    continue

                dh = hamming_distance(ht, prev_ht)
                
                # Filter: We only care if system changed (dh >= min_dh)
                # dh=0 means identical block root (empty block or no state change),
                # which is valid but boring for scaling laws.
                if dh >= min_dh:
                    sequences.append(seq)
                    n_values.append(nv)
                    delta_h_values.append(dh)
            
            prev_ht = ht

    if not n_values:
        print("Insufficient data after filtering.")
        return

    # --- Statistics ---
    n_array = np.array(n_values)
    dh_array = np.array(delta_h_values)
    
    avg_dh = np.mean(dh_array)
    std_dh = np.std(dh_array)
    median_dh = np.median(dh_array)
    min_val_dh = np.min(dh_array)
    max_val_dh = np.max(dh_array)
    
    # --- Scaling Law Fit ---
    # We fit: log(Delta H) = -beta * log(Nv) + C
    # => log(Delta H) = slope * log(Nv) + intercept
    # beta = -slope
    
    log_nv = np.log(n_array)
    log_dh = np.log(dh_array)
    
    slope, intercept = np.polyfit(log_nv, log_dh, 1)
    beta = -slope
    
    print("\n=== Hash State Drift Analysis (Exploratory) ===")
    print(f"Data points: {len(n_values)} transitions (from {row_count} blocks)")
    print(f"Filters: min_nv={min_nv}, min_dh={min_dh}")
    print(f"Range N_v: {np.min(n_array)} -> {np.max(n_array)}")
    print(f"Range dH:  {min_val_dh} -> {max_val_dh}")
    
    print("\n--- Statistics ---")
    print(f"Avg Delta H:    {avg_dh:.2f} bits (Expected ~128)")
    print(f"Std Delta H:    {std_dh:.2f}")
    print(f"Median Delta H: {median_dh:.2f}")
    
    print("\n--- Scaling Law Fit ---")
    print(f"Equation: Delta H ~ N_v ^ -beta")
    print(f"Beta:      {beta:.4f} (Expected ~0.0)")
    print(f"Intercept: {math.exp(intercept):.4f}")

    # Interpretation
    print("\n--- Interpretation ---")
    if abs(avg_dh - 128) > 20:
        print("[WARN] Average drift is far from 128. Possible bias or non-random hashing.")
    else:
        print("[PASS] Average drift consistent with random avalanche.")

    if abs(beta) > 0.1:
        print(f"[WARN] Non-zero scaling (beta={beta:.4f}). Potential pathology or artifact.")
    else:
        print("[PASS] Beta near zero. Confirming random walk behavior.")

    return n_array, dh_array, beta, avg_dh, intercept

def plot_results(n_array, dh_array, beta, avg_dh, intercept, output_path):
    """Generate and save the analysis plot."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not found, skipping plot generation.")
        return

    plt.figure(figsize=(10, 6))
    
    # Scatter plot
    plt.scatter(n_array, dh_array, alpha=0.5, s=10, label='Observed Drift')
    
    # Regression Line
    # log(y) = -beta * log(x) + C  =>  y = exp(C) * x^(-beta)
    x_line = np.linspace(min(n_array), max(n_array), 100)
    y_line = math.exp(intercept) * (x_line ** -beta)
    plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Fit (β={beta:.4f})')
    
    # Expected Line (Horizontal at 128)
    plt.axhline(y=128, color='green', linestyle='--', alpha=0.7, label='Ideal (128 bits)')
    
    plt.xscale('log')
    plt.yscale('linear') # Linear y-scale for bit count is usually easier to read around 128
    # But strictly for scaling law visual, log-log is standard. 
    # Given the range is [0, 256], linear Y is better to see the noise band.
    
    plt.ylim(0, 256)
    plt.xlabel('Verified Volume ($N_v$) [Log Scale]')
    plt.ylabel('Hash State Drift ($\Delta H$) [Bits]')
    plt.title('First Organism Hash State Drift Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    # Annotations
    stats_text = (
        f"$\\beta \\approx {beta:.4f}$\n"
        f"Avg $\\Delta H = {avg_dh:.1f}$ bits\n"
        f"$N = {len(n_array)}$ transitions"
    )
    plt.text(0.05, 0.05, stats_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze H_t scaling behavior.")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file from extract_ht_data.py")
    parser.add_argument("--min-nv", type=int, default=10, help="Minimum N_v to include in analysis (default: 10)")
    parser.add_argument("--min-dh", type=int, default=1, help="Minimum Delta H to include (default: 1)")
    parser.add_argument("--plot-output", "-p", help="Path to save the analysis plot (e.g., artifacts/figures/ht_delta_scaling.png)")
    
    args = parser.parse_args()
    
    try:
        results = analyze(args.input, args.min_nv, args.min_dh)
        if results and args.plot_output:
            n_array, dh_array, beta, avg_dh, intercept = results
            plot_results(n_array, dh_array, beta, avg_dh, intercept, args.plot_output)
    except Exception as e:
        sys.stderr.write(f"Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
