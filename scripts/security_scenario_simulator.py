# scripts/security_scenario_simulator.py
"""
GEMINI-K: SIGMA-III - "What-if" Security Scenario Simulator

This tool allows operators and governance teams to test the impact of
potential security posture changes without affecting live systems. It takes a
baseline posture, applies simulated modifications, and shows the resulting
change in security level and governance narrative.
"""

import argparse
import json
from pathlib import Path
from backend.security.posture import (
    SecurityPosture,
    summarize_security_for_governance,
    ReplayStatus,
    SeedClassification,
    LastMileValidation,
)

def main():
    parser = argparse.ArgumentParser(description="GEMINI-K 'What-if' Scenario Simulator")
    parser.add_argument(
        "--baseline-posture-file",
        type=Path,
        required=True,
        help="Path to a JSON file containing the baseline SecurityPosture."
    )
    # Arguments to simulate changes
    parser.add_argument("--set-replay-status", type=str, choices=["OK", "FAIL"])
    parser.add_argument("--set-seed-classification", type=str, choices=["PURE", "DRIFT"])
    parser.add_argument("--set-last-mile-validation", type=str, choices=["PASS", "FAIL"])
    
    args = parser.parse_args()

    # 1. Load baseline posture
    if not args.baseline_posture_file.exists():
        print(f"Error: Baseline posture file not found at '{args.baseline_posture_file}'", file=sys.stderr)
        exit(1)
        
    with open(args.baseline_posture_file, 'r') as f:
        try:
            baseline_posture: SecurityPosture = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in '{args.baseline_posture_file}'", file=sys.stderr)
            exit(1)

    # 2. Create the modified posture based on inputs
    modified_posture = baseline_posture.copy()
    
    if args.set_replay_status:
        modified_posture["replay_status"] = args.set_replay_status
    if args.set_seed_classification:
        modified_posture["seed_classification"] = args.set_seed_classification
    if args.set_last_mile_validation:
        modified_posture["last_mile_validation"] = args.set_last_mile_validation
        
    # 3. Get governance summaries for both postures
    baseline_summary = summarize_security_for_governance(baseline_posture)
    modified_summary = summarize_security_for_governance(modified_posture)
    
    # 4. Display the "what-if" report
    print("--- GEMINI-K: Security Scenario Simulation Report ---")
    
    print("\n[1] BASELINE POSTURE")
    print("--------------------")
    print(json.dumps(baseline_summary, indent=2))
    
    print("\n[2] SIMULATED POSTURE (What-if)")
    print("-----------------------------")
    print(json.dumps(modified_summary, indent=2))
    
    print("\n[3] IMPACT ANALYSIS")
    print("-------------------")
    if baseline_summary["security_level"] != modified_summary["security_level"]:
        print(f"IMPACT: Security level changed from '{baseline_summary['security_level']}' -> '{modified_summary['security_level']}'.")
        print(f"NARRATIVE: {modified_summary['narrative']}")
    else:
        print("IMPACT: No change in overall security level.")
        
    print("\n--- Simulation Complete ---")

if __name__ == "__main__":
    main()
