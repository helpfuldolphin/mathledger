# PHASE IV — REPLAY GOVERNANCE ORCHESTRATOR
"""
This script orchestrates the full replay governance analysis pipeline.
It connects the Phase III ledgering tools with the Phase IV analysis tools
to produce a single, comprehensive governance snapshot artifact.
"""
import argparse
import json
import os
import sys

# Add backend to path to allow direct imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.governance.replay_history import (
    build_replay_history,
    extract_replay_incidents,
    ReplayReceipt
)
from backend.governance.replay_analysis import (
    build_replay_governance_radar,
    evaluate_replay_for_promotion,
    build_replay_director_panel,
)

def read_receipts_from_jsonl(path: str) -> list[ReplayReceipt]:
    """Reads a jsonl file of replay receipts."""
    receipts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                receipts.append(ReplayReceipt(**data))
    return receipts

def main():
    """Main entry point for the orchestrator."""
    parser = argparse.ArgumentParser(description="Replay Governance Orchestrator")
    parser.add_argument("--receipts-file", required=True, help="Path to the input jsonl file of all replay receipts.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the snapshot artifact.")
    parser.add_argument("--criticality-config", default="config/replay_criticality_rules.yaml", help="Path to criticality rules config.")
    args = parser.parse_args()

    # --- Load Data ---
    try:
        receipts = read_receipts_from_jsonl(args.receipts_file)
    except FileNotFoundError:
        print(f"Error: Receipts file not found at {args.receipts_file}", file=sys.stderr)
        sys.exit(1)
    
    # --- Phase III: Build Foundational Artifacts ---
    # NOTE: The ReceiptIndexEntry is simplified here for orchestration
    ledger = build_replay_history({"receipts": receipts})
    incidents = extract_replay_incidents(receipts)

    # --- Phase IV: Build Analysis Artifacts ---
    radar = build_replay_governance_radar(ledger, incidents, criticality_config_path=args.criticality_config)
    
    # A global health summary would be built here from the ledger, but we'll use a placeholder
    # for this focused script.
    global_health_summary = {"status": "WARN" if incidents else "OK"}
    promotion_eval = evaluate_replay_for_promotion(radar, global_health_summary)
    director_panel = build_replay_director_panel(radar, promotion_eval)

    # --- Assemble Final Snapshot ---
    snapshot = {
        "radar": radar,
        "promotion_eval": promotion_eval,
        "director_panel": director_panel,
    }

    # --- Write Output ---
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "replay_governance_snapshot.json")
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(snapshot, f, indent=2, sort_keys=True)
        
    print(f"Successfully generated governance snapshot at {output_path}")

if __name__ == "__main__":
    main()
