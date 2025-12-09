#!/usr/bin/env python3
# PHASE IV — SUBSTRATE GOVERNANCE HARD GATE
#
# This script serves as a CI enforcement gate for substrate governance.
# It loads the latest Identity Ledger from an experiment run, analyzes it,
# and exits with a status code reflecting the substrate's promotion readiness.
#
# Exit Codes:
#   0: OK - Substrate is stable and conformant.
#   1: WARN - A non-blocking issue was detected.
#   2: BLOCK - A blocking issue (e.g., drift, side-effect) was detected.
#   10: ERROR - Script failed due to missing artifacts or other errors.

import argparse
import json
import sys
from pathlib import Path

# Add project root for backend imports
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.governance.analyzer import (
    analyze_substrate_identity_ledger,
    build_substrate_director_panel,
    evaluate_substrate_for_promotion,
)

def find_latest_ledger_artifact(artifacts_dir: Path) -> Path:
    """Finds the most recent manifest.json file in the directory."""
    manifests = list(artifacts_dir.glob("manifest_*.json"))
    if not manifests:
        raise FileNotFoundError(f"No manifest artifacts found in {artifacts_dir}")
    
    # Return the most recently modified file
    return max(manifests, key=lambda p: p.stat().st_mtime)

def main():
    """Main entry point for the governance check script."""
    parser = argparse.ArgumentParser(description="Substrate Governance Hard Gate Check")
    parser.add_argument(
        "--artifacts-dir",
        required=True,
        type=str,
        help="Directory containing the artifacts from a `run_uplift_u2.py` execution."
    )
    args = parser.parse_args()

    artifacts_path = Path(args.artifacts_dir)
    if not artifacts_path.is_dir():
        print(f"ERROR: Artifacts directory not found: {artifacts_path}", file=sys.stderr)
        sys.exit(10)

    try:
        # 1. Load the latest IdentityLedger artifact
        latest_manifest_path = find_latest_ledger_artifact(artifacts_path)
        print(f"INFO: Analyzing latest artifact: {latest_manifest_path.name}")
        with open(latest_manifest_path, "r") as f:
            manifest = json.load(f)
        
        identity_ledger = manifest.get("identity_ledger")
        if identity_ledger is None:
            print(f"ERROR: 'identity_ledger' field not found in manifest.", file=sys.stderr)
            sys.exit(10)

        # 2. Call the analysis pipeline
        # Note: The governance_summary is not needed for this check, so we pass an empty dict.
        identity_analysis = analyze_substrate_identity_ledger(identity_ledger)
        promotion_eval = evaluate_substrate_for_promotion(identity_analysis, {})
        director_panel = build_substrate_director_panel(identity_analysis, promotion_eval)

        print("\n--- Substrate Director Panel ---")
        print(json.dumps(director_panel, indent=2))
        print("------------------------------\n")

        # 3. Exit with the appropriate status code
        status = promotion_eval.get("status", "BLOCK")
        exit_code = 10 # Default to error
        if status == "OK":
            exit_code = 0
            print("VERDICT: Status is OK. Governance gate PASSED.")
        elif status == "WARN":
            exit_code = 1
            print(f"VERDICT: Status is WARN. Governance gate PASSED with warnings.")
            print("Reasons:", promotion_eval.get("blocking_reasons"))
        elif status == "BLOCK":
            exit_code = 2
            print(f"VERDICT: Status is BLOCK. Governance gate FAILED.")
            print("Blocking Reasons:", promotion_eval.get("blocking_reasons"))
            
        sys.exit(exit_code)

    except Exception as e:
        print(f"FATAL: An unexpected error occurred during governance check: {e}", file=sys.stderr)
        sys.exit(10)

if __name__ == "__main__":
    main()
