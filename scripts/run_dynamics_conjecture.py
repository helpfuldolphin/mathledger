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
CLI Runner for the Dynamics-Theory Unification Engine.
"""

import argparse
import json
import os
from analysis.conjecture_engine import run_conjecture_analysis

def load_jsonl(file_path: str) -> list:
    """Loads a JSONL file into a list of dictionaries."""
    records = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {file_path}: {e}")
        exit(1)
    return records

def main():
    """Main function to run the analysis from the command line."""
    parser = argparse.ArgumentParser(
        description="Run the Dynamics-Theory Unification Engine on experimental data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'baseline_path',
        type=str,
        help="Path to the baseline run's JSONL output file."
    )
    parser.add_argument(
        'rfl_path',
        type=str,
        help="Path to the RFL run's JSONL output file."
    )
    parser.add_argument(
        '--thresholds-config',
        type=str,
        default='config/dynamics_thresholds.json',
        help="Path to the dynamics thresholds configuration file."
    )
    parser.add_argument(
        '--slice-threshold',
        type=float,
        default=0.10,
        help="The preregistered uplift threshold (tau_i) for this slice."
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='artifacts/dynamics/conjecture_report.json',
        help="Path to save the final JSON report."
    )
    args = parser.parse_args()

    print("--- Running Dynamics-Theory Unification Engine ---")

    # Load data and configs
    print(f"Loading baseline data from: {args.baseline_path}")
    baseline_data = load_jsonl(args.baseline_path)
    print(f"Loading RFL data from: {args.rfl_path}")
    rfl_data = load_jsonl(args.rfl_path)
    print(f"Loading thresholds from: {args.thresholds_config}")
    try:
        with open(args.thresholds_config, 'r') as f:
            thresholds = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Threshold config not found at {args.thresholds_config}. Using defaults.")
        thresholds = {}

    conjectures_to_test = [
        "Conjecture 3.1",
        "Conjecture 4.1",
        "Conjecture 6.1",
        "Phase II Uplift"
    ]

    # Run the core analysis
    print("\nRunning conjecture analysis...")
    final_report = run_conjecture_analysis(
        baseline_records=baseline_data,
        rfl_records=rfl_data,
        slice_uplift_threshold=args.slice_threshold,
        conjectures_to_test=conjectures_to_test,
        thresholds=thresholds
    )

    # Emit the report
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output_path, 'w') as f:
        json.dump(final_report, f, indent=2)

    print(f"\nSuccessfully generated conjecture report at: {args.output_path}")
    print("\n--- Report Summary ---")
    print(json.dumps(final_report, indent=2))
    print("\n--- Engine Finished ---")


if __name__ == '__main__':
    main()
