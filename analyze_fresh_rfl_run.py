#!/usr/bin/env python3
"""
Analyze fresh RFL run and update documentation.
Run this after fo_rfl.jsonl is complete (1000 cycles).
"""
import json
import sys
from pathlib import Path
import pandas as pd

# Check if files exist
baseline_path = Path("results/fo_baseline.jsonl")
rfl_path = Path("results/fo_rfl.jsonl")

if not baseline_path.exists():
    print(f"ERROR: {baseline_path} not found")
    sys.exit(1)

if not rfl_path.exists():
    print(f"ERROR: {rfl_path} not found")
    sys.exit(1)

# Count lines
baseline_lines = sum(1 for _ in open(baseline_path))
rfl_lines = sum(1 for _ in open(rfl_path))

print(f"Baseline: {baseline_lines} cycles")
print(f"RFL: {rfl_lines} cycles")

if rfl_lines < 1000:
    print(f"WARNING: RFL run incomplete ({rfl_lines}/1000 cycles)")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)

# Load and analyze
print("\nLoading data...")
baseline_data = []
rfl_data = []

with open(baseline_path) as f:
    for line in f:
        if line.strip():
            baseline_data.append(json.loads(line))

with open(rfl_path) as f:
    for line in f:
        if line.strip():
            rfl_data.append(json.loads(line))

df_baseline = pd.DataFrame(baseline_data)
df_rfl = pd.DataFrame(rfl_data)

# Calculate abstention rates
baseline_abstention = df_baseline['abstention'].mean()
rfl_abstention = df_rfl['abstention'].mean()
delta = rfl_abstention - baseline_abstention

# Post burn-in (cycles 200-1000)
burn_in = 200
df_baseline_post = df_baseline[df_baseline['cycle'] >= burn_in]
df_rfl_post = df_rfl[df_rfl['cycle'] >= burn_in]

baseline_post_abstention = df_baseline_post['abstention'].mean()
rfl_post_abstention = df_rfl_post['abstention'].mean()
delta_post = rfl_post_abstention - baseline_post_abstention

# Check first 50-100 cycles for anomalies
print("\n" + "=" * 60)
print("ANALYSIS RESULTS")
print("=" * 60)

print(f"\nOverall Abstention Rates:")
print(f"  Baseline: {baseline_abstention:.3f} ({baseline_abstention*100:.1f}%)")
print(f"  RFL:      {rfl_abstention:.3f} ({rfl_abstention*100:.1f}%)")
print(f"  Δ:        {delta:+.3f} ({delta*100:+.1f}%)")

print(f"\nPost Burn-in (cycles {burn_in}-{len(df_rfl)-1}):")
print(f"  Baseline: {baseline_post_abstention:.3f} ({baseline_post_abstention*100:.1f}%)")
print(f"  RFL:      {rfl_post_abstention:.3f} ({rfl_post_abstention*100:.1f}%)")
print(f"  Δ:        {delta_post:+.3f} ({delta_post*100:+.1f}%)")

# Check first 50-100 cycles
print(f"\nFirst 50 cycles:")
df_rfl_first50 = df_rfl[df_rfl['cycle'] < 50]
rfl_first50_abstention = df_rfl_first50['abstention'].mean()
print(f"  RFL abstention: {rfl_first50_abstention:.3f} ({rfl_first50_abstention*100:.1f}%)")

print(f"\nFirst 100 cycles:")
df_rfl_first100 = df_rfl[df_rfl['cycle'] < 100]
rfl_first100_abstention = df_rfl_first100['abstention'].mean()
print(f"  RFL abstention: {rfl_first100_abstention:.3f} ({rfl_first100_abstention*100:.1f}%)")

# Check for anomalies
print(f"\nAnomalies Check:")
anomalies = []

# Check if all cycles abstained
if df_rfl['abstention'].all():
    anomalies.append("All RFL cycles abstained (100% abstention rate)")

# Check if no cycles abstained
if not df_rfl['abstention'].any():
    anomalies.append("No RFL cycles abstained (0% abstention rate)")

# Check if abstention increases over time (bad sign)
first_half = df_rfl[df_rfl['cycle'] < 500]['abstention'].mean()
second_half = df_rfl[df_rfl['cycle'] >= 500]['abstention'].mean()
if second_half > first_half + 0.05:  # 5% increase
    anomalies.append(f"Abstention increased in second half ({first_half:.3f} → {second_half:.3f})")

# Check H_t diversity
h_t_values = df_rfl['roots'].apply(lambda x: x['h_t'] if isinstance(x, dict) else x).tolist()
unique_h_t = len(set(h_t_values))
if unique_h_t < 10:
    anomalies.append(f"Low H_t diversity: only {unique_h_t} unique values in 1000 cycles")

if anomalies:
    print("  ⚠️  Anomalies detected:")
    for anomaly in anomalies:
        print(f"    - {anomaly}")
else:
    print("  ✅ No major anomalies detected")

# Check cycle range
print(f"\nCycle Range:")
print(f"  Baseline: {df_baseline['cycle'].min()} to {df_baseline['cycle'].max()}")
print(f"  RFL:      {df_rfl['cycle'].min()} to {df_rfl['cycle'].max()}")

# Check roots presence
print(f"\nRoots Present:")
baseline_has_roots = all('roots' in row and 'h_t' in row.get('roots', {}) for _, row in df_baseline.iterrows())
rfl_has_roots = all('roots' in row and 'h_t' in row.get('roots', {}) for _, row in df_rfl.iterrows())
print(f"  Baseline: {'✅' if baseline_has_roots else '❌'}")
print(f"  RFL:      {'✅' if rfl_has_roots else '❌'}")

print("\n" + "=" * 60)
print("Summary for FO_RUN_SUMMARY.md:")
print("=" * 60)
print(f"| results\\fo_baseline.jsonl | {baseline_lines} | {baseline_abstention:.3f} | Pass | {df_baseline['cycle'].min()} to {df_baseline['cycle'].max()} | {'Pass' if baseline_has_roots else 'Fail'} |")
print(f"| results\\fo_rfl.jsonl | {rfl_lines} | {rfl_abstention:.3f} | Pass | {df_rfl['cycle'].min()} to {df_rfl['cycle'].max()} | {'Pass' if rfl_has_roots else 'Fail'} |")

