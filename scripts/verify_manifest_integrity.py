# PHASE II — NOT RUN IN PHASE I
"""
Verifies the cryptographic bindings and schema of a U2 Manifest file.
"""
import argparse
import hashlib
import json
import sys
import yaml
import os

def get_prereg_hash(prereg_file_path, experiment_id):
    """Computes the hash of a specific preregistration entry."""
    with open(prereg_file_path, 'r', encoding='utf-8') as f:
        prereg_data = yaml.safe_load(f)
    experiment_entry = next((item for item in prereg_data if item["experiment_id"] == experiment_id), None)
    if not experiment_entry:
        raise ValueError(f"Experiment ID '{experiment_id}' not found in prereg file.")
    canonical_entry_yaml = yaml.dump(experiment_entry, sort_keys=True, default_flow_style=False)
    return hashlib.sha256(canonical_entry_yaml.encode('utf-8')).hexdigest()

def get_file_hash(file_path):
    """Computes the SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def verify_manifest(manifest_path, prereg_file, slice_config, results_file):
    """
    Verifies the manifest file.
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except FileNotFoundError:
        return {"status": "ERROR", "message": f"Manifest file not found: {manifest_path}", "exit_code": 1}
    except json.JSONDecodeError:
        return {"status": "ERROR", "message": "Invalid JSON in manifest file.", "exit_code": 1}

    # 1. Verify schema
    required_fields = [
        "manifest_schema_version", "preregistration_hash", "slice_config_hash",
        "code_version_hash", "deterministic_seed", "results_hash", "experiment_id"
    ]
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        return {"status": "ERROR", "message": f"Manifest is missing required fields: {missing_fields}", "exit_code": 1}

    experiment_id = manifest["experiment_id"]

    # 2. Verify preregistration hash
    try:
        expected_prereg_hash = get_prereg_hash(prereg_file, experiment_id)
        if manifest["preregistration_hash"] != expected_prereg_hash:
            return {"status": "ERROR", "message": "Preregistration hash mismatch.", "exit_code": 2}
    except Exception as e:
        return {"status": "ERROR", "message": f"Could not compute prereg hash: {e}", "exit_code": 2}

    # 3. Verify slice config hash
    expected_slice_hash = get_file_hash(slice_config)
    if not expected_slice_hash:
         return {"status": "ERROR", "message": f"Slice config file not found: {slice_config}", "exit_code": 3}
    if manifest["slice_config_hash"] != expected_slice_hash:
        return {"status": "ERROR", "message": "Slice config hash mismatch.", "exit_code": 3}

    # 4. Verify results hash
    expected_results_hash = get_file_hash(results_file)
    if not expected_results_hash:
         return {"status": "ERROR", "message": f"Results file not found: {results_file}", "exit_code": 4}
    if manifest["results_hash"] != expected_results_hash:
        return {"status": "ERROR", "message": "Results hash mismatch.", "exit_code": 4}


    return {
        "status": "SUCCESS",
        "manifest_path": manifest_path,
        "verified_bindings": [
            "preregistration",
            "slice_config",
            "results"
        ],
        "exit_code": 0
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verifies a U2 Manifest file.")
    parser.add_argument("--manifest", required=True, help="Path to the U2 Manifest JSON file.")
    parser.add_argument("--prereg-file", required=True, help="Path to the preregistration YAML file.")
    parser.add_argument("--slice-config", required=True, help="Path to the slice configuration file.")
    parser.add_argument("--results-file", required=True, help="Path to the results output file.")
    parser.add_argument("--json", action="store_true", help="Output result as a JSON object.")
    args = parser.parse_args()

    result = verify_manifest(args.manifest, args.prereg_file, args.slice_config, args.results_file)

    if args.json:
        output = {k: v for k, v in result.items() if k != 'exit_code'}
        print(json.dumps(output, indent=2))
    else:
        print(result.get("message", f"Status: {result.get('status')}"))

    sys.exit(result.get("exit_code", 1))

if __name__ == "__main__":
    main()
