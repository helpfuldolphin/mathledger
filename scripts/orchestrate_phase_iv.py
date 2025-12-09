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
Phase IV Pipeline Orchestrator (orchestrate_phase_iv.py)
=========================================================

This script drives the full, end-to-end analysis pipeline as specified in
the Phase IV architecture. It simulates a complete experimental run, from
data generation to final governance reporting.

Author: Gemini M, Dynamics-Theory Unification Analyst
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

# Ensure the analysis module is in the path
sys.path.append(os.getcwd())
from analysis.conjecture_engine import generate_mock_data

def run_command(command: list):
    """Helper to run a command and check for errors."""
    print(f"\n>>> EXECUTING: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"::error:: Command failed with exit code {result.returncode}")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)
    print(result.stdout)
    return result

def main():
    parser = argparse.ArgumentParser(description="Run the full Phase IV analysis pipeline.")
    parser.add_argument(
        '--run-id',
        type=str,
        default=f"mock_run_{int(datetime.now().timestamp())}",
        help="A unique ID for this experimental run."
    )
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['positive_logistic', 'null', 'instability', 'degenerate'],
        default='positive_logistic',
        help="The mock data scenario to generate for this run."
    )
    args = parser.parse_args()
    
    run_id = args.run_id
    scenario = args.scenario
    python_executable = os.path.join(".venv", "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(".venv", "bin", "python")


    # --- Step 0: Setup Directories ---
    results_dir = os.path.join('results', run_id)
    snapshot_dir = 'artifacts/snapshots'
    os.makedirs(os.path.join(results_dir, 'baseline'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'rfl'), exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    # --- Step 1: Generate Mock Data ---
    # (In a real pipeline, this step would be the actual experiment)
    print(f"--- Generating mock data for scenario '{scenario}' ---")
    baseline_path = os.path.join(results_dir, 'baseline', 'run.jsonl')
    rfl_path = os.path.join(results_dir, 'rfl', 'run.jsonl')
    
    baseline_records, rfl_records = generate_mock_data(scenario)
    with open(baseline_path, 'w') as f:
        for record in baseline_records: f.write(json.dumps(record) + '\n')
    with open(rfl_path, 'w') as f:
        for record in rfl_records: f.write(json.dumps(record) + '\n')
    print(f"Mock data generated in {results_dir}")

    # --- Step 2: Run Single-Run Conjecture Analysis ---
    print("\n--- Running single-run conjecture analysis ---")
    conjecture_report_path = os.path.join(results_dir, 'conjecture_report.json')
    dynamics_cmd = [
        python_executable,
        'scripts/run_dynamics_conjecture.py',
        baseline_path,
        rfl_path,
        '--output-path', conjecture_report_path
    ]
    run_command(dynamics_cmd)

    # --- Step 3: Persist Snapshot ---
    print("\n--- Persisting snapshot for historical analysis ---")
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    snapshot_filename = f"{timestamp}_{run_id}_snapshot.json"
    snapshot_path = os.path.join(snapshot_dir, snapshot_filename)
    os.rename(conjecture_report_path, snapshot_path)
    print(f"Snapshot saved to {snapshot_path}")

    # --- Step 4: Update Global State ---
    print("\n--- Updating global governance state ---")
    governance_cmd = [
        python_executable,
        'scripts/update_global_state.py',
        '--latest-snapshot', snapshot_path
    ]
    run_command(governance_cmd)
    
    print(f"\n--- Pipeline for run '{run_id}' completed successfully! ---")
    print("Final reports available in: artifacts/governance/")

if __name__ == '__main__':
    main()
