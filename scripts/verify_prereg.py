# PHASE II — NOT RUN IN PHASE I
"""
Verifies that a given experiment ID has a valid and locked entry in the
preregistration YAML file.
"""
import argparse
import hashlib
import json
import sys
import yaml

def verify_preregistration(prereg_file_path, experiment_id):
    """
    Validates a preregistration entry.

    Args:
        prereg_file_path (str): Path to the preregistration YAML file.
        experiment_id (str): The experiment ID to verify.

    Returns:
        A dictionary with the validation result.
    """
    try:
        with open(prereg_file_path, 'r', encoding='utf-8') as f:
            prereg_data = yaml.safe_load(f)
            if not isinstance(prereg_data, list):
                return {"status": "ERROR", "message": "Preregistration file is not a list of experiments.", "exit_code": 1}

            # Find the specific experiment block
            experiment_entry = None
            for entry in prereg_data:
                if entry.get("experiment_id") == experiment_id:
                    experiment_entry = entry
                    break
            
            if not experiment_entry:
                 return {"status": "ERROR", "message": f"Experiment ID '{experiment_id}' not found.", "exit_code": 2}

    except FileNotFoundError:
        return {"status": "ERROR", "message": f"Preregistration file not found: {prereg_file_path}", "exit_code": 1}
    except yaml.YAMLError as e:
        return {"status": "ERROR", "message": f"Invalid YAML in preregistration file: {e}", "exit_code": 1}

    required_fields = ['description', 'slice_config', 'slice_config_hash', 'seed', 'success_metrics']
    missing_fields = [field for field in required_fields if field not in experiment_entry]

    if missing_fields:
        return {"status": "ERROR", "message": f"Experiment '{experiment_id}' is missing required fields: {missing_fields}", "exit_code": 3}

    # Hash the canonical YAML representation of the specific experiment block
    # This ensures the hash is stable regardless of file ordering or comments
    canonical_entry_yaml = yaml.dump(experiment_entry, sort_keys=True, default_flow_style=False)
    prereg_hash = hashlib.sha256(canonical_entry_yaml.encode('utf-8')).hexdigest()

    return {
        "status": "SUCCESS",
        "experiment_id": experiment_id,
        "preregistration_hash": prereg_hash,
        "details": {
            "slice_config": experiment_entry.get("slice_config"),
            "slice_config_hash": experiment_entry.get("slice_config_hash"),
            "seed": experiment_entry.get("seed")
        },
        "exit_code": 0
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Verifies a preregistration entry.")
    parser.add_argument("--prereg-file", required=True, help="Path to the preregistration YAML file.")
    parser.add_argument("--experiment-id", required=True, help="The unique ID of the experiment to verify.")
    parser.add_argument("--json", action="store_true", help="Output result as a JSON object.")
    args = parser.parse_args()

    result = verify_preregistration(args.prereg_file, args.experiment_id)

    if args.json:
        # Don't include exit_code in the JSON output itself
        output = {k: v for k, v in result.items() if k != 'exit_code'}
        print(json.dumps(output, indent=2))
    else:
        print(result.get("message", f"Status: {result.get('status')}"))

    sys.exit(result.get("exit_code", 1))

if __name__ == "__main__":
    main()
