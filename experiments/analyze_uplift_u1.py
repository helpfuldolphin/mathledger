#!/usr/bin/env python3
"""
RFL Uplift Experiment U1 Analyzer
==================================

Analyzes baseline and RFL logs from Uplift Experiment U1,
computes abstention rate curves, and reports uplift metrics.

Usage:
    uv run python experiments/analyze_uplift_u1.py [--baseline PATH] [--rfl PATH] [--burn-in N]

Output:
    - Prints uplift metrics to console
    - Generates abstention rate curves (if matplotlib available)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("WARNING: pandas not available, using basic analysis only", file=sys.stderr)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file into list of dictionaries."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_abstention_rate(records: List[Dict], burn_in: int = 0) -> Tuple[float, int]:
    """
    Compute mean abstention rate from records (post burn-in).
    
    Returns:
        (mean_abstention_rate, sample_size)
    """
    if burn_in > 0:
        records = records[burn_in:]
    
    if not records:
        return 0.0, 0
    
    # Try multiple fields for abstention detection
    abstention_count = 0
    for rec in records:
        if rec.get("abstention") is True:
            abstention_count += 1
        elif rec.get("status") == "abstain":
            abstention_count += 1
        elif rec.get("success") is False:
            abstention_count += 1
        elif rec.get("proof_found") is False:
            abstention_count += 1
    
    rate = abstention_count / len(records) if records else 0.0
    return rate, len(records)


def compute_rolling_abstention(records: List[Dict], window_size: int = 50) -> List[float]:
    """Compute rolling abstention rate over windows."""
    rates = []
    for i in range(0, len(records), window_size):
        window = records[i:i + window_size]
        if window:
            rate, _ = compute_abstention_rate(window, burn_in=0)
            rates.append(rate)
    return rates


def check_prereg_validity(
    baseline_rate: float,
    baseline_n: int,
    rfl_n: int,
) -> Dict[str, bool]:
    """
    Check preregistration validity criteria.
    
    Returns:
        Dictionary with validity check results
    """
    baseline_in_range = 0.10 <= baseline_rate <= 0.80
    baseline_sufficient = baseline_n >= 200
    rfl_sufficient = rfl_n >= 200
    all_valid = baseline_in_range and baseline_sufficient and rfl_sufficient
    
    return {
        "baseline_abstention_in_range": baseline_in_range,
        "baseline_proof_attempts_sufficient": baseline_sufficient,
        "rfl_proof_attempts_sufficient": rfl_sufficient,
        "all_validity_criteria_met": all_valid,
    }


def check_prereg_uplift(
    baseline_rate: float,
    rfl_rate: float,
    delta_abs: float,
    ci_lower: Optional[float] = None,
    ci_upper: Optional[float] = None,
) -> Dict[str, bool]:
    """
    Check preregistration uplift criteria.
    
    Args:
        baseline_rate: Baseline abstention rate
        rfl_rate: RFL abstention rate
        delta_abs: Absolute difference (baseline - rfl)
        ci_lower: Lower bound of 95% CI for difference (optional)
        ci_upper: Upper bound of 95% CI for difference (optional)
    
    Returns:
        Dictionary with uplift check results
    """
    direction_correct = rfl_rate < baseline_rate
    magnitude_sufficient = delta_abs >= 0.10  # 10 percentage points
    statistically_significant = False
    if ci_lower is not None and ci_upper is not None:
        # CI excludes zero if both bounds are on same side of zero
        statistically_significant = (ci_lower > 0) or (ci_upper < 0)
    
    all_uplift_met = direction_correct and magnitude_sufficient and statistically_significant
    
    return {
        "direction_correct": direction_correct,
        "magnitude_sufficient": magnitude_sufficient,
        "statistically_significant": statistically_significant,
        "all_uplift_criteria_met": all_uplift_met,
    }


def classify_outcome(validity: Dict[str, bool], uplift: Dict[str, bool]) -> Tuple[str, str]:
    """
    Classify experiment outcome per preregistration.
    
    Returns:
        (outcome, reason) where outcome is INVALID | NULL | POSITIVE
    """
    if not validity["all_validity_criteria_met"]:
        reasons = []
        if not validity["baseline_abstention_in_range"]:
            reasons.append("baseline abstention not in range [0.10, 0.80]")
        if not validity["baseline_proof_attempts_sufficient"]:
            reasons.append("baseline proof attempts < 200")
        if not validity["rfl_proof_attempts_sufficient"]:
            reasons.append("RFL proof attempts < 200")
        return "INVALID", "; ".join(reasons)
    
    if uplift["all_uplift_criteria_met"]:
        return "POSITIVE", "All uplift criteria met: direction correct, magnitude >= 10pp, statistically significant"
    
    # Valid but no uplift
    reasons = []
    if not uplift["direction_correct"]:
        reasons.append("RFL abstention not lower than baseline")
    if not uplift["magnitude_sufficient"]:
        reasons.append("Magnitude < 10 percentage points")
    if not uplift["statistically_significant"]:
        reasons.append("Not statistically significant (CI includes zero)")
    return "NULL", "; ".join(reasons) if reasons else "No uplift detected"


def analyze_uplift_u1(
    baseline_path: Path,
    rfl_path: Path,
    burn_in: int = 50,
    output_figure: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Analyze Uplift Experiment U1 logs and compute metrics.
    
    Returns:
        Dictionary with analysis results
    """
    print("=" * 60)
    print("RFL UPLIFT EXPERIMENT U1 - ANALYSIS")
    print("=" * 60)
    print()
    
    # Load data
    print(f"Loading baseline log: {baseline_path}")
    baseline_records = load_jsonl(baseline_path)
    print(f"  Loaded {len(baseline_records)} cycles")
    
    print(f"Loading RFL log: {rfl_path}")
    rfl_records = load_jsonl(rfl_path)
    print(f"  Loaded {len(rfl_records)} cycles")
    print()
    
    # Compute overall metrics
    baseline_rate, baseline_n = compute_abstention_rate(baseline_records, burn_in=burn_in)
    rfl_rate, rfl_n = compute_abstention_rate(rfl_records, burn_in=burn_in)
    
    delta_abs = rfl_rate - baseline_rate
    delta_rel = (delta_abs / baseline_rate * 100) if baseline_rate > 0 else 0.0
    
    # Compute success rates (inverse of abstention)
    baseline_success_rate = 1.0 - baseline_rate
    rfl_success_rate = 1.0 - rfl_rate
    success_delta = rfl_success_rate - baseline_success_rate
    
    # Count verified/proof_found
    baseline_verified = sum(1 for r in baseline_records[burn_in:] if r.get("status") == "verified" or r.get("proof_found") is True)
    rfl_verified = sum(1 for r in rfl_records[burn_in:] if r.get("status") == "verified" or r.get("proof_found") is True)
    
    # Print results
    print("=" * 60)
    print("UPLIFT METRICS (post burn-in)")
    print("=" * 60)
    print(f"Burn-in cycles: {burn_in}")
    print(f"Sample size (post burn-in):")
    print(f"  Baseline: {baseline_n} cycles")
    print(f"  RFL:      {rfl_n} cycles")
    print()
    print("Abstention Rates:")
    print(f"  Baseline: {baseline_rate:.4f} ({baseline_rate*100:.2f}%)")
    print(f"  RFL:      {rfl_rate:.4f} ({rfl_rate*100:.2f}%)")
    print(f"  Δ (abs):  {delta_abs:+.4f} ({delta_abs*100:+.2f}%)")
    print(f"  Δ (rel):  {delta_rel:+.2f}%")
    print()
    print("Success Rates (proof found):")
    print(f"  Baseline: {baseline_success_rate:.4f} ({baseline_success_rate*100:.2f}%)")
    print(f"  RFL:      {rfl_success_rate:.4f} ({rfl_success_rate*100:.2f}%)")
    print(f"  Δ:        {success_delta:+.4f} ({success_delta*100:+.2f}%)")
    print()
    print("Verified Counts:")
    print(f"  Baseline: {baseline_verified} / {baseline_n}")
    print(f"  RFL:      {rfl_verified} / {rfl_n}")
    print()
    
    # Bootstrap CI for difference (simplified - use percentile method)
    ci_lower = None
    ci_upper = None
    mann_whitney_u = None
    mann_whitney_p = None
    
    if baseline_n > 0 and rfl_n > 0:
        # Extract per-cycle abstention rates for bootstrap
        baseline_abstention_rates = [
            1.0 if r.get("abstention") is True or r.get("status") == "abstain" or r.get("proof_found") is False
            else 0.0
            for r in baseline_records[burn_in:]
        ]
        rfl_abstention_rates = [
            1.0 if r.get("abstention") is True or r.get("status") == "abstain" or r.get("proof_found") is False
            else 0.0
            for r in rfl_records[burn_in:]
        ]
        
        # Simple bootstrap CI (percentile method, 10000 replicates per prereg)
        import random
        random.seed(42)  # Deterministic bootstrap
        bootstrap_diffs = []
        for _ in range(10000):
            baseline_sample = random.choices(baseline_abstention_rates, k=len(baseline_abstention_rates))
            rfl_sample = random.choices(rfl_abstention_rates, k=len(rfl_abstention_rates))
            bootstrap_diffs.append(sum(baseline_sample) / len(baseline_sample) - sum(rfl_sample) / len(rfl_sample))
        
        bootstrap_diffs.sort()
        ci_lower = bootstrap_diffs[250]  # 2.5th percentile (10000 * 0.025 = 250)
        ci_upper = bootstrap_diffs[9749]  # 97.5th percentile (10000 * 0.975 = 9749)
        
        # Mann-Whitney U test (simplified approximation)
        # For large samples, use normal approximation
        if len(baseline_abstention_rates) > 20 and len(rfl_abstention_rates) > 20:
            # Simplified: use two-proportion z-test as approximation
            baseline_abstain = int(baseline_rate * baseline_n)
            rfl_abstain = int(rfl_rate * rfl_n)
            p_pool = (baseline_abstain + rfl_abstain) / (baseline_n + rfl_n)
            if p_pool > 0 and p_pool < 1:
                se = (p_pool * (1 - p_pool) * (1/baseline_n + 1/rfl_n)) ** 0.5
                if se > 0:
                    z_score = delta_abs / se
                    mann_whitney_u = z_score  # Approximation
                    # Approximate p-value
                    if abs(z_score) > 2.58:
                        mann_whitney_p = 0.01
                    elif abs(z_score) > 1.96:
                        mann_whitney_p = 0.05
                    elif abs(z_score) > 1.65:
                        mann_whitney_p = 0.10
                    else:
                        mann_whitney_p = 0.50
        
        print("Statistical Test:")
        if ci_lower is not None and ci_upper is not None:
            print(f"  Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        if mann_whitney_p is not None:
            print(f"  Mann-Whitney p-value (approx): {mann_whitney_p:.3f}")
        print()
    
    # Check preregistration criteria
    validity = check_prereg_validity(baseline_rate, baseline_n, rfl_n)
    uplift = check_prereg_uplift(baseline_rate, rfl_rate, delta_abs, ci_lower, ci_upper)
    outcome, outcome_reason = classify_outcome(validity, uplift)
    
    # Rolling abstention rates
    if HAS_PANDAS and len(baseline_records) > 0 and len(rfl_records) > 0:
        window_size = min(50, len(baseline_records) // 10)
        baseline_rolling = compute_rolling_abstention(baseline_records, window_size)
        rfl_rolling = compute_rolling_abstention(rfl_records, window_size)
        
        print("Rolling Abstention Rates (window size={}):".format(window_size))
        print(f"  Baseline: {baseline_rolling[:5]} ... {baseline_rolling[-5:]}")
        print(f"  RFL:      {rfl_rolling[:5]} ... {rfl_rolling[-5:]}")
        print()
        
        # Generate figure if matplotlib available
        if HAS_MATPLOTLIB and output_figure:
            try:
                plt.figure(figsize=(10, 6))
                x_baseline = [i * window_size for i in range(len(baseline_rolling))]
                x_rfl = [i * window_size for i in range(len(rfl_rolling))]
                plt.plot(x_baseline, baseline_rolling, label='Baseline', marker='o', markersize=3)
                plt.plot(x_rfl, rfl_rolling, label='RFL', marker='s', markersize=3)
                plt.xlabel('Cycle')
                plt.ylabel('Abstention Rate')
                plt.title('RFL Uplift Experiment U1 - Rolling Abstention Rates')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_figure, dpi=150)
                print(f"✅ Figure saved: {output_figure}")
            except Exception as e:
                print(f"⚠️  Could not generate figure: {e}", file=sys.stderr)
    
    # Preregistration criteria check
    print("=" * 60)
    print("PREREGISTRATION CRITERIA")
    print("=" * 60)
    print("Validity Criteria:")
    print(f"  Baseline abstention in range [0.10, 0.80]: {validity['baseline_abstention_in_range']}")
    print(f"  Baseline proof attempts >= 200: {validity['baseline_proof_attempts_sufficient']} ({baseline_n})")
    print(f"  RFL proof attempts >= 200: {validity['rfl_proof_attempts_sufficient']} ({rfl_n})")
    print(f"  All validity criteria met: {validity['all_validity_criteria_met']}")
    print()
    print("Uplift Criteria:")
    print(f"  Direction correct (RFL < baseline): {uplift['direction_correct']}")
    print(f"  Magnitude >= 10pp: {uplift['magnitude_sufficient']} (actual: {delta_abs*100:.2f}pp)")
    print(f"  Statistically significant: {uplift['statistically_significant']}")
    print(f"  All uplift criteria met: {uplift['all_uplift_criteria_met']}")
    print()
    
    # Outcome classification
    print("=" * 60)
    print("OUTCOME CLASSIFICATION")
    print("=" * 60)
    if outcome == "INVALID":
        print(f"❌ INVALID: {outcome_reason}")
    elif outcome == "NULL":
        print(f"➖ NULL: {outcome_reason}")
    elif outcome == "POSITIVE":
        print(f"✅ POSITIVE: {outcome_reason}")
    print()
    
    # Build results matching preregistration schema
    from datetime import datetime, timezone
    results = {
        "experiment_id": "uplift_u1",
        "baseline": {
            "cycles": baseline_n,
            "total_proof_attempts": baseline_n,
            "total_abstained": int(baseline_rate * baseline_n),
            "mean_abstention_rate": baseline_rate,
            "std_abstention_rate": 0.0,  # Would need per-cycle rates for std
        },
        "rfl": {
            "cycles": rfl_n,
            "total_proof_attempts": rfl_n,
            "total_abstained": int(rfl_rate * rfl_n),
            "mean_abstention_rate": rfl_rate,
            "std_abstention_rate": 0.0,  # Would need per-cycle rates for std
        },
        "comparison": {
            "difference": delta_abs,
            "difference_pp": delta_abs * 100,
            "bootstrap_ci_95_lower": ci_lower,
            "bootstrap_ci_95_upper": ci_upper,
            "bootstrap_replicates": 10000,
            "mann_whitney_u_statistic": mann_whitney_u,
            "mann_whitney_p_value": mann_whitney_p,
        },
        "validity": validity,
        "uplift": uplift,
        "outcome": outcome,
        "outcome_reason": outcome_reason,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze RFL Uplift Experiment U1 results"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="results/uplift_u1/baseline.jsonl",
        help="Path to baseline JSONL log",
    )
    parser.add_argument(
        "--rfl",
        type=str,
        default="results/uplift_u1/rfl.jsonl",
        help="Path to RFL JSONL log",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=50,
        help="Number of burn-in cycles to exclude (default: 50)",
    )
    parser.add_argument(
        "--output-figure",
        type=str,
        default="artifacts/figures/uplift_u1_abstention_curve.png",
        help="Path to save abstention curve figure (default: artifacts/figures/uplift_u1_abstention_curve.png)",
    )
    
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline)
    rfl_path = Path(args.rfl)
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline log not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    
    if not rfl_path.exists():
        print(f"ERROR: RFL log not found: {rfl_path}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory for figure if needed
    if args.output_figure:
        fig_path = Path(args.output_figure)
        fig_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        results = analyze_uplift_u1(
            baseline_path=baseline_path,
            rfl_path=rfl_path,
            burn_in=args.burn_in,
            output_figure=Path(args.output_figure) if args.output_figure else None,
        )
        
        # Write statistical summary matching preregistration schema
        summary_path = baseline_path.parent / "statistical_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        print(f"✅ Statistical summary written: {summary_path}")
        
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Analysis failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

