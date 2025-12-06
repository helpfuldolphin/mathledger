# EXPERIMENTAL: Analysis for Imperfect Verifier Simulation

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

def analyze_files(file_paths: List[Path]):
    # Data structure: epsilon -> stats
    # stats: total, fp, fn, gt_true, gt_false
    results: Dict[float, Dict[str, int]] = defaultdict(lambda: {
        "total": 0, "fp": 0, "fn": 0, "gt_true": 0, "gt_false": 0
    })

    for p in file_paths:
        if not p.exists():
            print(f"Warning: File {p} does not exist, skipping.")
            continue
        
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    eps = float(data["epsilon"])
                    kind = data["kind"]
                    gt_verified = data["ground_truth_verified"]
                    
                    results[eps]["total"] += 1
                    if gt_verified:
                        results[eps]["gt_true"] += 1
                    else:
                        results[eps]["gt_false"] += 1
                        
                    if kind == "fp":
                        results[eps]["fp"] += 1
                    elif kind == "fn":
                        results[eps]["fn"] += 1
                        
                except json.JSONDecodeError:
                    continue

    # Print Table
    print("\n=== Imperfect Verifier Error Rates ===")
    print(f"{ 'Epsilon':<10} | {'Samples':<10} | {'FP Rate':<10} | {'FN Rate':<10} | {'Total Err':<10}")
    print("-" * 60)
    
    sorted_eps = sorted(results.keys())
    for eps in sorted_eps:
        stat = results[eps]
        gt_true = stat["gt_true"]
        gt_false = stat["gt_false"]
        total = stat["total"]
        
        fp_rate = stat["fp"] / gt_false if gt_false > 0 else 0.0
        fn_rate = stat["fn"] / gt_true if gt_true > 0 else 0.0
        total_err = (stat["fp"] + stat["fn"]) / total if total > 0 else 0.0
        
        print(f"{eps:<10.4f} | {total:<10} | {fp_rate:<10.4f} | {fn_rate:<10.4f} | {total_err:<10.4f}")
    print("======================================\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze Imperfect Verifier Results")
    parser.add_argument("files", nargs="+", type=Path, help="Input JSONL files")
    args = parser.parse_args()
    
    analyze_files(args.files)

if __name__ == "__main__":
    main()
