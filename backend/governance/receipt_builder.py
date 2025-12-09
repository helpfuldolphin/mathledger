# PHASE II — NOT RUN IN PHASE I
"""
Constructs a deterministic Governance Receipt from the outputs of the
verification scripts. This is the v2 builder, compliant with the G3 replay gate.
"""
import json
import hashlib
from datetime import datetime, timezone

GOVERNANCE_VERSION = "GOVERNANCE-2.0.0"

def generate_deterministic_timestamp(results_hash):
    """
    Generates an ISO 8601 timestamp based on a hash.

    This ensures that the receipt is perfectly reproducible. We use the
    results_hash as the source of entropy.
    """
    timestamp_seed = int(results_hash[:8], 16)
    dt = datetime.fromtimestamp(timestamp_seed, tz=timezone.utc)
    return dt.isoformat()

def build_governance_receipt(prereg_result, manifest_result, uplift_gate_result, manifest_data):
    """
    Builds the v2 Governance Receipt JSON object.

    Args:
        prereg_result (dict): The successful JSON output from verify_prereg.py.
        manifest_result (dict): The successful JSON output from verify_manifest_integrity.py.
        uplift_gate_result (dict): The successful JSON output from verify_uplift_gates.py.
        manifest_data (dict): The content of the manifest.json file itself.

    Returns:
        A dictionary representing the governance_receipt.json content.
    """
    results_hash = manifest_data.get("results_hash")
    if not results_hash:
        raise ValueError("Manifest data must contain a 'results_hash' for deterministic timestamp generation.")

    # Extract replay verification data from the G3 gate's successful output
    replay_verification = uplift_gate_result.get("replay_verification")
    if not replay_verification or replay_verification.get("status") != "VERIFIED_CONSISTENT":
        raise ValueError("Uplift gate result is missing a successful replay verification.")

    receipt = {
        "receipt_version": GOVERNANCE_VERSION,
        "experiment_id": uplift_gate_result["experiment_id"],
        "manifest_path": manifest_result["manifest_path"],
        "verification_timestamp": generate_deterministic_timestamp(results_hash),
        "governance_record": {
            "prereg_state": {
                "status": "VERIFIED",
                "preregistration_hash": prereg_result["preregistration_hash"]
            },
            "manifest_state": {
                "status": "VERIFIED",
                "code_version_hash": manifest_data["code_version_hash"],
                "deterministic_seed": manifest_data["deterministic_seed"]
            },
            "hash_state": {
                "status": "VERIFIED",
                "verified_hashes": {
                    "slice_config": manifest_data["slice_config_hash"],
                    "results": results_hash,
                    "ht_binding": manifest_data["ht_binding_hash"]
                }
            },
            "integrity_state": {
                "status": "VERIFIED",
                "verified_bindings": manifest_result["verified_bindings"]
            },
            "replay_state": {
                "status": "VERIFIED_CONSISTENT",
                "original_results_hash": replay_verification["results_hash"],
                "replay_ht_binding_hash": replay_verification["ht_binding_hash"]
            },
            "final_decision": {
                "decision": "admissible",
                "message": "All Phase II governance gates passed, including G3 Determinism Replay."
            }
        }
    }
    # This now references a future v2 schema definition
    receipt['$schema'] = 'docs/U2_GOVERNANCE_PIPELINE_V2.md#5-canonical-governance-receipt-json-schema'

    return receipt

def main():
    """A simple CLI for testing the v2 receipt builder."""
    import argparse
    parser = argparse.ArgumentParser(description="Build a v2 Governance Receipt")
    parser.add_argument("--prereg-json", required=True)
    parser.add_argument("--manifest-integrity-json", required=True)
    parser.add_argument("--uplift-gates-json", required=True)
    parser.add_argument("--manifest-json", required=True)
    args = parser.parse_args()

    with open(args.prereg_json, 'r') as f: prereg = json.load(f)
    with open(args.manifest_integrity_json, 'r') as f: manifest_integrity = json.load(f)
    with open(args.uplift_gates_json, 'r') as f: uplift_gates = json.load(f)
    with open(args.manifest_json, 'r') as f: manifest = json.load(f)

    receipt = build_governance_receipt(prereg, manifest_integrity, uplift_gates, manifest)
    print(json.dumps(receipt, indent=2))

if __name__ == '__main__':
    main()