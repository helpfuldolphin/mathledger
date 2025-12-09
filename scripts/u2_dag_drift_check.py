# scripts/u2_dag_drift_check.py
"""
PHASE III - CI Pattern: Structural Drift Check Gate.

This script compares a new DAG posture snapshot against a previous (e.g., 'main' branch)
snapshot and determines if the structural drift is acceptable.

It serves as a CI gate to prevent unacceptable regressions in the DAG structure.
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root for local imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.dag.posture_analysis import evaluate_dag_drift_acceptability

EXIT_CODE_OK = 0
EXIT_CODE_BLOCKED = 1
EXIT_CODE_ERROR = 2

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Structural Drift Check CI Gate.")
    parser.add_argument(
        "new_posture_path",
        type=Path,
        help="Path to the new DAG posture snapshot JSON file to be evaluated."
    )
    parser.add_argument(
        "--previous-posture-path",
        type=Path,
        required=True,
        help="Path to the previous or 'main' branch DAG posture snapshot to compare against."
    )
    args = parser.parse_args()

    # --- Load Postures ---
    try:
        with open(args.previous_posture_path, 'r') as f:
            old_posture = json.load(f)
        with open(args.new_posture_path, 'r') as f:
            new_posture = json.load(f)
    except FileNotFoundError as e:
        print(f"[ERROR] Posture file not found: {e}", file=sys.stderr)
        sys.exit(EXIT_CODE_ERROR)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in posture file: {e}", file=sys.stderr)
        sys.exit(EXIT_CODE_ERROR)

    # --- Evaluate Drift ---
    print(f"Comparing new posture '{args.new_posture_path.name}' against previous '{args.previous_posture_path.name}'...")
    result = evaluate_dag_drift_acceptability(old_posture, new_posture)
    
    status = result["drift_status"]
    reasons = result["reasons"]
    
    # --- Report and Exit ---
    print(f"\nDRIFT STATUS: [{status}]")
    print("\nReasons:")
    for reason in reasons:
        print(f"- {reason}")

    # Print detailed comparison for context
    print("\nDetailed Comparison:")
    print(json.dumps(result["comparison"], indent=2))

    if status == "BLOCKED":
        print("\nGATE: FAILED. Unacceptable structural drift detected.", file=sys.stderr)
        sys.exit(EXIT_CODE_BLOCKED)
    elif status == "WARN":
        print("\nGATE: PASSED WITH WARNINGS. Please review the changes carefully.", file=sys.stderr)
        sys.exit(EXIT_CODE_OK)
    else: # OK
        print("\nGATE: PASSED. Drift is within acceptable parameters.", file=sys.stderr)
        sys.exit(EXIT_CODE_OK)

if __name__ == "__main__":
    main()
