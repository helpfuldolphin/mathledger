# Copyright 2025 MathLedger
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Global State Update Script (update_global_state.py)
===================================================

This script updates the project's global state by incorporating the latest
experimental analysis. It generates the historical conjecture timeline and the
final uplift readiness report.

Author: Gemini M, Dynamics-Theory Unification Analyst
"""
import argparse
import json
import os
from typing import List, Dict

from analysis.governance import build_conjecture_timeline, combine_conjectures_with_governance

# --- Data Layer ---

def query_snapshots(snapshot_dir: str, limit: int = 50) -> List[Dict]:
    """
    Queries the file-based snapshot store and returns an ordered list
    of conjecture reports, as specified in the Phase IV architecture.
    """
    try:
        all_snapshot_files = sorted([
            f for f in os.listdir(snapshot_dir) if f.endswith('_snapshot.json')
        ])
    except FileNotFoundError:
        return []

    snapshots_to_load = all_snapshot_files[-limit:]
    
    loaded_data = []
    for filename in snapshots_to_load:
        filepath = os.path.join(snapshot_dir, filename)
        with open(filepath, 'r') as f:
            try:
                data = json.load(f)
                parts = filename.split('_')
                data['_meta'] = {'timestamp': parts[0] + '_' + parts[1], 'run_id': '_'.join(parts[2:-1])}
                loaded_data.append(data)
            except (json.JSONDecodeError, IndexError):
                print(f"Warning: Skipping corrupted or malformed snapshot '{filename}'.")
                continue
    return loaded_data

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Update global dynamics and governance state.")
    parser.add_argument(
        '--latest-snapshot',
        type=str,
        required=True,
        help="Path to the latest conjecture report snapshot to be incorporated."
    )
    parser.add_argument(
        '--snapshot-dir',
        type=str,
        default='artifacts/snapshots',
        help="Directory containing all historical snapshots."
    )
    parser.add_argument(
        '--governance-posture',
        type=str,
        default='config/governance_posture.json',
        help="Path to the current governance posture file."
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts/governance',
        help="Directory to save the output reports."
    )
    args = parser.parse_args()

    # --- Load all necessary inputs ---
    print(f"Loading latest snapshot from: {args.latest_snapshot}")
    try:
        with open(args.latest_snapshot, 'r') as f:
            latest_snapshot = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"::error::Failed to load latest snapshot: {e}")
        exit(1)

    print(f"Loading governance posture from: {args.governance_posture}")
    try:
        with open(args.governance_posture, 'r') as f:
            governance_posture = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"::error::Failed to load governance posture: {e}")
        exit(1)
        
    print(f"Querying historical snapshots from: {args.snapshot_dir}")
    all_snapshots = query_snapshots(args.snapshot_dir)
    # Ensure the current run's snapshot is included if it's already there
    if not any(s['_meta']['run_id'] in args.latest_snapshot for s in all_snapshots):
         all_snapshots.append(latest_snapshot)

    # --- Run Governance and Timeline Logic ---
    print("Building conjecture timeline...")
    timeline_report = build_conjecture_timeline(all_snapshots)
    
    print("Combining with governance posture...")
    readiness_report = combine_conjectures_with_governance(governance_posture, latest_snapshot)

    # --- Emit Artifacts ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    timeline_path = os.path.join(args.output_dir, 'conjecture_timeline.json')
    readiness_path = os.path.join(args.output_dir, 'uplift_readiness_report.json')

    print(f"Writing timeline report to: {timeline_path}")
    with open(timeline_path, 'w') as f:
        json.dump(timeline_report, f, indent=2)

    print(f"Writing readiness report to: {readiness_path}")
    with open(readiness_path, 'w') as f:
        json.dump(readiness_report, f, indent=2)
        
    print("\nGlobal state update complete.")
    print(f"Uplift Readiness Flag: {readiness_report['uplift_readiness_flag']}")
    print(f"Dynamics Status: {readiness_report['dynamics_status']}")


if __name__ == '__main__':
    main()
