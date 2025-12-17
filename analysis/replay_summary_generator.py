# EXPERIMENTAL — NOT GOVERNANCE, NOT GATING
"""
Aggregates component-level replay metrics into a single statistical summary.
This tool is for advisory and data collection purposes ONLY. It does not
render verdicts or implement any gating logic.
"""
import argparse
import datetime
import json
import os
import sys
import glob
import statistics
from typing import Dict, List, Any

COMPONENT_SCHEMA_VERSION = "1.0"
SUMMARY_SCHEMA_VERSION = "1.0"

def validate_component_data(data: Dict[str, Any], filepath: str) -> bool:
    """Validates a single component's data against the contract."""
    if data.get("schema_version") != COMPONENT_SCHEMA_VERSION:
        print(f"Warning: Skipping {filepath}. Invalid schema_version.", file=sys.stderr)
        return False
    required_keys = ["name", "determinism_rate", "drift_metric"]
    if not all(key in data for key in required_keys):
        print(f"Warning: Skipping {filepath}. Missing one of {required_keys}.", file=sys.stderr)
        return False
    return True

def calculate_stats(data: List[float]) -> Dict[str, Any]:
    """Calculates a set of descriptive statistics for a list of numbers."""
    if not data:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}

    std_dev = statistics.stdev(data) if len(data) > 1 else 0

    return {
        "mean": round(statistics.mean(data), 4),
        "median": round(statistics.median(data), 4),
        "min": round(min(data), 4),
        "max": round(max(data), 4),
        "std_dev": round(std_dev, 4)
    }

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Replay Statistical Summary Generator")
    parser.add_argument("--input-dir", required=True, help="Directory containing ComponentMetric JSON files.")
    parser.add_argument("--output-file", default="./replay_run_summary.json", help="Path to write the summary file.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    components = []
    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if validate_component_data(data, filepath):
                # Only keep raw data, no verdict fields
                components.append({
                    "name": data["name"],
                    "determinism_rate": data["determinism_rate"],
                    "drift_metric": data["drift_metric"]
                })
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON file: {filepath}", file=sys.stderr)
            continue

    # Deterministic ordering for consistency
    components.sort(key=lambda x: x['name'])

    # Extract data for statistical analysis
    determinism_rates = [c['determinism_rate'] for c in components]
    drift_metrics = [c['drift_metric'] for c in components]

    run_id = os.environ.get("GITHUB_RUN_ID", f"local-run-{int(datetime.datetime.utcnow().timestamp())}")
    summary: Dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": "ANALYSIS",
        "scope_note": "NOT_GOVERNANCE_NOT_GATING",
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "summary_statistics": {
            "component_count": len(components),
            "determinism_rate": calculate_stats(determinism_rates),
            "drift_metric": calculate_stats(drift_metrics)
        },
        "components": components
    }

    output_json = json.dumps(summary, indent=2)

    try:
        with open(args.output_file, 'w') as f:
            f.write(output_json)
        print(f"Replay summary written to {args.output_file}")
    except IOError as e:
        print(f"Error: Could not write to output file {args.output_file}: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
