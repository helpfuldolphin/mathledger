"""
PHASE III — LIVE METRIC GOVERNANCE
CI Conformance Gating CLI
"""
import argparse
import json
import os
import sys
import glob
from typing import Any, Dict, List

# Add project root to path to allow importing the governance module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiments.metric_governance import can_promote_metric, load_promotion_policy

def gate_family(args):
    """
    Handler for the 'gate-family' subcommand.
    
    Loads all candidate snapshots for a family, compares them against their
    baselines according to policy, and exits with a non-zero status if any
    metric fails its promotion gate.
    """
    if not args.json_output:
        print(f"--- Running Conformance Gate for Family: '{args.family_name}' ---")
    
    policy = load_promotion_policy()
    
    # Find all candidate snapshots
    candidate_paths = glob.glob(os.path.join(args.candidate_dir, f"{args.family_name}*.json"))
    
    if not candidate_paths:
        result_summary = {
            "family_name": args.family_name,
            "status": "WARNING",
            "message": f"No candidate snapshots found for family '{args.family_name}' in {args.candidate_dir}",
            "metrics": []
        }
        if args.json_output:
            print(json.dumps(result_summary, indent=2))
        else:
            print(result_summary["message"])
        sys.exit(0) # Not a failure if no metrics found
        
    if not args.json_output:
        print(f"Found {len(candidate_paths)} candidate metrics to check.")
    
    results = []
    has_blocking_failures = False
    
    for cand_path in candidate_paths:
        metric_filename = os.path.basename(cand_path)
        metric_name = metric_filename.replace(".json", "") # Extract metric name
        base_path = os.path.join(args.baseline_dir, metric_filename)
        
        if not os.path.exists(base_path):
            result = {
                "metric_name": metric_name,
                "status": "SKIPPED",
                "reason": f"No baseline snapshot found for '{metric_name}' at {base_path}."
            }
            results.append(result)
            if not args.json_output:
                print(f"  [WARN] {metric_filename}: {result['reason']}")
            continue
            
        # Load snapshots
        with open(cand_path, 'r') as f:
            candidate_snapshot = json.load(f)
        with open(base_path, 'r') as f:
            baseline_snapshot = json.load(f)
            
        # Apply the promotion gate
        can_promote, reason = can_promote_metric(baseline_snapshot, candidate_snapshot, policy)
        
        result = {
            "metric_name": metric_name,
            "status": "PASS" if can_promote else "FAIL",
            "reason": reason,
            "conformance_level": candidate_snapshot["levels"]["L2_domain_coverage"]["status"] # Example level
        }
        results.append(result)
        
        if not can_promote:
            has_blocking_failures = True
            if not args.json_output:
                print(f"  [FAIL] {metric_filename}: {reason}")
        else:
            if not args.json_output:
                print(f"  [PASS] {metric_filename}: {reason}")
            
    # Final verdict
    final_status = "SUCCESS"
    final_message = "All metrics in the family passed the promotion gate."
    exit_code = 0
    
    if has_blocking_failures:
        final_status = "FAILURE"
        final_message = "Some metrics failed the promotion gate."
        exit_code = 1
    
    result_summary = {
        "family_name": args.family_name,
        "status": final_status,
        "message": final_message,
        "metrics": results
    }

    if args.json_output:
        print(json.dumps(result_summary, indent=2))
    else:
        print(f"\n{final_status}: {final_message}")

    if not args.dry_run and exit_code != 0:
        sys.exit(exit_code)
    elif args.dry_run and exit_code != 0:
        if not args.json_output:
            print(f"Dry-run: Would have exited with code {exit_code}")
        sys.exit(0) # Always exit 0 in dry-run mode
    else:
        sys.exit(0) # All good, or dry-run was successful

def main():
    parser = argparse.ArgumentParser(description="Metric Conformance Governance CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # --- gate-family subcommand ---
    gate_parser = subparsers.add_parser("gate-family", help="Gate a family of metrics against a promotion policy.")
    gate_parser.add_argument("family_name", help="The name of the metric family (e.g., 'uplift_u2').")
    gate_parser.add_argument("baseline_dir", help="Directory containing baseline metric snapshots.")
    gate_parser.add_argument("candidate_dir", help="Directory containing candidate metric snapshots.")
    gate_parser.add_argument("--dry-run", action="store_true", help="Perform checks but always exit with code 0.")
    gate_parser.add_argument("--json-output", action="store_true", help="Output results as JSON to stdout.")
    gate_parser.set_defaults(func=gate_family)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()