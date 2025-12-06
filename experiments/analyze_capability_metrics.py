import argparse
import sys
import os
import pandas as pd
import numpy as np
from typing import List, Dict

# Ensure root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfl.experiment import RFLExperiment
from substrate.security.runtime_env import get_database_url

def run_trials(
    executor: RFLExperiment,
    mode_name: str,
    trials: int,
    steps: int,
    breadth: int,
    total: int
) -> List[Dict]:
    results = []
    for i in range(trials):
        run_id = f"cap_{{mode_name.lower()}}_{i+1}"
        print(f"Running {mode_name} trial {i+1}/{trials}...")
        
        # Note: In a real RFL scenario, we would pass policy_context or specific flags
        # to enable the learned policy. For now, we run the standard engine.
        # If 'mode_name' == 'RFL', we might expect a policy to be injected here.
        res = executor.run(
            run_id=run_id,
            derive_steps=steps,
            max_breadth=breadth,
            max_total=total
        )
        
        results.append({
            "Mode": mode_name,
            "Trial": i + 1,
            "Proofs/Hr": res.throughput_proofs_per_hour,
            "Max Depth": res.max_depth,
            "Abstention %": res.abstention_rate * 100.0,
            "Success %": res.success_rate * 100.0
        })
    return results

def main():
    parser = argparse.ArgumentParser(description="Analyze Capability Metrics")
    parser.add_argument("--budget", type=int, default=500, help="Max statements per run")
    parser.add_argument("--trials", type=int, default=3, help="Trials per mode")
    args = parser.parse_args()

    db_url = get_database_url()
    executor = RFLExperiment(db_url=db_url, system_id=1)

    print(f"Starting Capability Analysis (Budget={args.budget}, Trials={args.trials})")
    print("-" * 60)

    # 1. Baseline Run (Standard BFS/Random)
    baseline_data = run_trials(executor, "Baseline", args.trials, steps=50, breadth=100, total=args.budget)

    # 2. RFL Run (Simulated/Placeholder for now)
    # TODO: Inject policy here once CLI supports --policy via RFLExperiment
    rfl_data = run_trials(executor, "RFL", args.trials, steps=50, breadth=100, total=args.budget)

    # Combine and Analyze
    all_data = baseline_data + rfl_data
    df = pd.DataFrame(all_data)

    print("\nRaw Data:")
    print(df.to_string(index=False, float_format="%.2f"))

    # Aggregation
    summary = df.groupby("Mode")[["Proofs/Hr", "Max Depth", "Abstention %"]].agg(['mean', 'std'])
    
    print("\nSummary (Mean +/- Std):")
    print(summary.to_string(float_format="%.2f"))

    # Uplift Calculation
    means = df.groupby("Mode").mean(numeric_only=True)
    if "Baseline" in means.index and "RFL" in means.index:
        base_t = means.loc["Baseline", "Proofs/Hr"]
        rfl_t = means.loc["RFL", "Proofs/Hr"]
        uplift = (rfl_t / base_t - 1.0) * 100.0 if base_t > 0 else 0.0
        print(f"\nRFL Throughput Uplift: {{uplift:+.2f}}%")

if __name__ == "__main__":
    main()