# scripts/collect_telemetry_snapshot.py
#
# ORCHESTRATOR: Telemetry Conformance Snapshot Collection
# JURISDICTION: File-based Telemetry Ingestion, Artifact Generation
# IDENTITY: GEMINI H, Telemetry Sentinel

import argparse
import json
import sys
import os
from pathlib import Path

# Ensure backend modules can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.telemetry.snapshot_builder import build_telemetry_conformance_snapshot

def main():
    """
    Main entry point for collecting a telemetry snapshot from a JSONL log file.
    """
    parser = argparse.ArgumentParser(
        description="Reads a telemetry log file (.jsonl) and generates a conformance snapshot artifact."
    )
    parser.add_argument(
        "source_log",
        type=str,
        help="Path to the source telemetry log file (e.g., 'cycles.jsonl')."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/telemetry",
        help="Directory to save the output snapshot file."
    )
    args = parser.parse_args()

    source_path = Path(args.source_log)
    output_path = Path(args.output_dir)

    print(f"--- Starting Telemetry Snapshot Collection for: {source_path} ---")

    if not source_path.is_file():
        print(f"[ERROR] Source log file not found: {source_path}", file=sys.stderr)
        sys.exit(2)

    try:
        # 1. Load raw events from the JSONL file
        raw_events = []
        with open(source_path, 'r') as f:
            for line in f:
                if line.strip():
                    raw_events.append(json.loads(line))
        print(f"Loaded {len(raw_events)} raw events from source log.")

        # 2. Build the conformance snapshot using the core module
        snapshot = build_telemetry_conformance_snapshot(raw_events)
        print("Successfully built conformance snapshot.")

        # 3. Save the snapshot artifact
        output_path.mkdir(parents=True, exist_ok=True)
        snapshot_file = output_path / "telemetry_snapshot.json"

        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
        
        print(f"\nSnapshot artifact successfully generated at: {snapshot_file}")
        
        # 4. Print snapshot summary to stdout for CI logs
        print("\n--- SNAPSHOT SUMMARY ---")
        summary_view = {k: v for k, v in snapshot.items() if k != 'critical_violations'}
        summary_view['num_critical_violations'] = len(snapshot['critical_violations'])
        print(json.dumps(summary_view, indent=2, sort_keys=True))
        print("--- END SUMMARY ---")
        
        sys.exit(0)

    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse JSONL file. Invalid JSON on a line: {e}", file=sys.stderr)
        sys.exit(3)
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred during snapshot collection: {e}", file=sys.stderr)
        sys.exit(4)

if __name__ == "__main__":
    # Add datetime import for snapshot_builder
    import datetime
    main()
