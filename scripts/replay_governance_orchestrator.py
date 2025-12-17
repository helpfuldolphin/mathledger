# scripts/replay_governance_orchestrator.py
"""
Reads component-level replay metrics, applies governance rules,
and generates a final replay_governance_snapshot.json artifact.

This script operates under the rules defined in:
docs/system_law/replay/REPLAY_ORCHESTRATOR_CONTRACT.md
"""
import argparse
import datetime
import json
import os
import sys
import glob
from typing import Dict, List, Any, Optional

# Constants based on the contract
COMPONENT_SCHEMA_VERSION = "1.0"
SNAPSHOT_SCHEMA_VERSION = "1.0"

# Hardcoded rules, can be moved to a config file later.
# For now, this is simpler for PR-1.
# Let's read the yaml file as per the blueprint
import yaml

def load_rules(config_path: str = "replay_criticality_rules.yaml") -> Dict[str, float]:
    """Loads governance thresholds from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            rules = yaml.safe_load(f)
        return rules.get("thresholds", {})
    except (IOError, yaml.YAMLError) as e:
        print(f"Error loading rules config: {e}", file=sys.stderr)
        # Return default high-tolerance values on error to avoid blocking
        return {
            "min_determinism_rate": 99.5,
            "max_drift_metric": 0.10
        }

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

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Replay Governance Orchestrator")
    parser.add_argument("--input-dir", required=True, help="Directory containing ComponentMetric JSON files.")
    parser.add_argument("--output-file", default="./replay_governance_snapshot.json", help="Path to write the governance snapshot.")
    parser.add_argument("--dry-run", action="store_true", help="Run evaluation but do not write output file.")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory not found at {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    rules = load_rules()
    min_determinism_rate = rules.get("min_determinism_rate", 99.5)
    max_drift_metric = rules.get("max_drift_metric", 0.10)

    components = []
    json_files = glob.glob(os.path.join(args.input_dir, '*.json'))

    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            if validate_component_data(data, filepath):
                components.append(data)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON file: {filepath}", file=sys.stderr)
            continue

    # Deterministic ordering as per contract
    components.sort(key=lambda x: x['name'])

    run_id = os.environ.get("GITHUB_RUN_ID", f"local-run-{int(datetime.datetime.utcnow().timestamp())}")
    snapshot: Dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "artifact_version": "1.0.0", # Legacy field from original spec
        "run_id": run_id,
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "radar_status": "", # To be populated
        "determinism_rate": 0.0,
        "promotion_eval": {
            "verdict": "promotion_ok",
            "reasons": []
        },
        "components": []
    }

    if not components:
        snapshot["radar_status"] = "UNSTABLE"
        snapshot["promotion_eval"]["verdict"] = "BLOCK"
        snapshot["promotion_eval"]["reasons"].append("no component data found")
    else:
        # Aggregation
        total_determinism = sum(c['determinism_rate'] for c in components)
        avg_determinism = total_determinism / len(components)
        snapshot["determinism_rate"] = round(avg_determinism, 2)

        # Evaluation
        if snapshot["determinism_rate"] < min_determinism_rate:
            snapshot["promotion_eval"]["verdict"] = "BLOCK"
            snapshot["promotion_eval"]["reasons"].append(
                f"Determinism rate ({snapshot['determinism_rate']:.2f}%) is below threshold ({min_determinism_rate}%)"
            )

        processed_components = []
        for c in components:
            is_blocking = c['drift_metric'] > max_drift_metric
            if is_blocking:
                snapshot["promotion_eval"]["verdict"] = "BLOCK"
                snapshot["promotion_eval"]["reasons"].append(
                    f"Component '{c['name']}' drift metric ({c['drift_metric']}) exceeds ceiling ({max_drift_metric})"
                )
            processed_components.append({
                "name": c["name"],
                "determinism_rate": c["determinism_rate"],
                "drift_metric": c["drift_metric"],
                "is_blocking": is_blocking
            })
        snapshot["components"] = processed_components

        if snapshot["promotion_eval"]["verdict"] == "promotion_ok":
            snapshot["radar_status"] = "STABLE"
            snapshot["promotion_eval"]["reasons"].append("All metrics are within governance thresholds.")
        else:
            snapshot["radar_status"] = "UNSTABLE"

    # Output
    output_json = json.dumps(snapshot, indent=2)

    if args.dry_run:
        print("--- DRY RUN MODE ---")
        print(output_json)
    else:
        try:
            with open(args.output_file, 'w') as f:
                f.write(output_json)
            print(f"Governance snapshot written to {args.output_file}")
        except IOError as e:
            print(f"Error: Could not write to output file {args.output_file}: {e}", file=sys.stderr)
            sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    main()
