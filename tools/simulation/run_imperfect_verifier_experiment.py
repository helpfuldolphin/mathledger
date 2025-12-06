# EXPERIMENTAL: Imperfect verifier simulation
# SAFETY: NEVER use this in the production pipeline.

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to sys.path to ensure imports work
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from derivation.verification import StatementVerifier
from derivation.bounds import SliceBounds
from tools.simulation.noisy_verifier import NoisyVerifierWrapper

# A small corpus of formulas
TAUTOLOGIES = [
    "p -> p",
    "((p -> q) /\ p) -> q",
    "p \/ ~p",
    "~(p /\ ~p)",
    "p -> (q -> p)",
    "(p -> q) -> ((q -> r) -> (p -> r))",
    "~~p -> p",
    "p -> (p \/ q)",
]

NON_TAUTOLOGIES = [
    "p",
    "q",
    "p -> q",
    "q -> p",
    "p /\ ~p",
    "~(p -> p)",
    "(p -> q) -> (q -> p)",
    "p \/ q",
    "p /\ q",
]

def run_experiment(epsilon: float, n_samples: int, output_path: Path, seed: int):
    print(f"Starting Imperfect Verifier Experiment")
    print(f"  Epsilon:   {epsilon}")
    print(f"  Samples:   {n_samples}")
    print(f"  Output:    {output_path}")
    print(f"  Seed:      {seed}")

    # Initialize Verifiers
    bounds = SliceBounds()
    real_verifier = StatementVerifier(bounds)
    noisy_verifier = NoisyVerifierWrapper(real_verifier, epsilon=epsilon, seed=seed)

    rng = random.Random(seed)
    
    # Counters for summary
    stats = {
        "total": 0,
        "flipped": 0,
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
        "ground_truth_true": 0,
        "ground_truth_false": 0
    }

    all_formulas = TAUTOLOGIES + NON_TAUTOLOGIES

    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(n_samples):
            statement = rng.choice(all_formulas)
            
            # 1. Ground Truth
            real_outcome = real_verifier.verify(statement)
            is_tautology = real_outcome.verified
            
            # 2. Noisy Verdict
            noisy_outcome = noisy_verifier.verify(statement)
            noisy_verdict = noisy_outcome.verified
            
            # 3. Analyze
            is_flipped = (is_tautology != noisy_verdict)
            
            kind = "unknown"
            if is_tautology and noisy_verdict:
                kind = "tp" # True Positive
            elif not is_tautology and not noisy_verdict:
                kind = "tn" # True Negative
            elif not is_tautology and noisy_verdict:
                kind = "fp" # False Positive
            elif is_tautology and not noisy_verdict:
                kind = "fn" # False Negative

            # Update stats
            stats["total"] += 1
            if is_flipped:
                stats["flipped"] += 1
            
            if is_tautology:
                stats["ground_truth_true"] += 1
            else:
                stats["ground_truth_false"] += 1
                
            if kind == "tp": stats["true_positive"] += 1
            if kind == "tn": stats["true_negative"] += 1
            if kind == "fp": stats["false_positive"] += 1
            if kind == "fn": stats["false_negative"] += 1

            # Write record
            record = {
                "statement": statement,
                "epsilon": epsilon,
                "ground_truth_verified": is_tautology,
                "noisy_verified": noisy_verdict,
                "is_flipped": is_flipped,
                "kind": kind,
                "sample_index": i
            }
            f.write(json.dumps(record) + "\n")
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples...")

    # Print Summary
    print("\n--- Experiment Summary ---")
    print(f"Total Samples: {stats['total']}")
    print(f"Flip Count:    {stats['flipped']} ({stats['flipped']/stats['total']*100:.2f}%)")
    
    gt_true = stats["ground_truth_true"]
    gt_false = stats["ground_truth_false"]
    
    fn_rate = stats["false_negative"] / gt_true if gt_true > 0 else 0.0
    fp_rate = stats["false_positive"] / gt_false if gt_false > 0 else 0.0
    
    print(f"False Negative Rate (among True stmts): {fn_rate:.4f}")
    print(f"False Positive Rate (among False stmts): {fp_rate:.4f}")
    print(f"Approx Error Rate (FN+FP / Total): {(stats['false_negative'] + stats['false_positive'])/stats['total']:.4f}")
    print("--------------------------")

def main():
    parser = argparse.ArgumentParser(description="Run Imperfect Verifier Simulation")
    parser.add_argument("--epsilon", type=float, required=True, help="Verifier bias epsilon (0.0 to 0.5)")
    parser.add_argument("--n-samples", type=int, default=10000, help="Number of samples to run")
    parser.add_argument("--output", type=Path, required=True, help="Path to output JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    try:
        run_experiment(args.epsilon, args.n_samples, args.output, args.seed)
    except Exception as e:
        print(f"Error running experiment: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
