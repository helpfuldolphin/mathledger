#!/usr/bin/env python3
"""Generate LaTeX ablation table rows for whitepaper v5."""

import csv
import os

def short(x):
    """Truncate string to 8 characters."""
    return x[:8] if x else "—"

def fmt(n):
    """Format number with 2 decimal places."""
    if n in ("", "—", None):
        return "—"
    try:
        return f"{float(n):.2f}"
    except (ValueError, TypeError):
        return "—"

def main():
    """Generate LaTeX table rows from baseline and guided runs CSV."""
    baseline_file = "artifacts/wpv5/baseline_runs.csv"
    guided_file = "artifacts/wpv5/guided_runs.csv"
    outf = "artifacts/wpv5/ablation_rows.tex"

    lines = []

    # Read baseline runs
    if os.path.exists(baseline_file):
        try:
            with open(baseline_file, newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    name = f"BL-{i+1}"
                    line = (f"{name} & \\texttt{{—}} & {row['slice']} & {fmt(row['cpu_hours'])} & "
                            f"{row.get('beam_k', '—')} & {fmt(row['proofs_per_hour'])} & "
                            f"{fmt(row['median_verify_ms'])} & {row['depth']} & "
                            f"{fmt(row['abstain_pct'])} & \\texttt{{{short(row['block_root_short'])}}} \\\\")
                    lines.append(line)
        except Exception as e:
            print(f"Error reading baseline CSV: {e}")
    else:
        print(f"Warning: Baseline file {baseline_file} not found")

    # Read guided runs
    if os.path.exists(guided_file):
        try:
            with open(guided_file, newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    name = f"G-{i+1}"
                    policy_hash = short(row.get('policy_hash', ''))
                    line = (f"{name} & \\texttt{{{policy_hash}}} & {row['slice']} & {fmt(row['cpu_hours'])} & "
                            f"{row.get('beam_k', '—')} & {fmt(row['proofs_per_hour'])} & "
                            f"{fmt(row['median_verify_ms'])} & {row['depth']} & "
                            f"{fmt(row['abstain_pct'])} & \\texttt{{{short(row['block_root_short'])}}} \\\\")
                    lines.append(line)
        except Exception as e:
            print(f"Error reading guided CSV: {e}")
    else:
        print(f"Warning: Guided file {guided_file} not found")

    if not lines:
        print("No data found in any CSV file")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(outf), exist_ok=True)

    # Write LaTeX file
    try:
        with open(outf, "w") as f:
            f.write("\n".join(lines))
        print(f"WROTE {outf}")
        print(f"Generated {len(lines)} table rows:")
        for line in lines:
            print(f"  {line}")
    except Exception as e:
        print(f"Error writing LaTeX file: {e}")

if __name__ == "__main__":
    main()
