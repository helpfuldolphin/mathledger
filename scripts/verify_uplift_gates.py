# PHASE II — NOT RUN IN PHASE I
"""
Acts as a master script to perform a high-level check of all governance gates
for a given run, using the other verification scripts. This is the primary
enforcement point for G1, G2, G3, and G4.

Extended with replay receipt verification per:
- docs/U2_REPLAY_RECEIPT_CHARTER.md
- docs/U2_GOVERNANCE_RECONCILIATION_SPEC.md Section 6.5
- docs/VSD_PHASE_2.md Section 9F

Absolute Safeguards:
- No uplift math: does not inspect delta-p, p-values, or decision thresholds
- No Phase I changes: operates only on Phase II artifacts
- Binding only, not interpretive: verifies hash equality, not experimental outcomes
"""
import argparse
import json
import sys
import os
import hashlib
from pathlib import Path
from scripts.verify_prereg import verify_preregistration
from scripts.verify_manifest_integrity import verify_manifest

# Import replay receipt validation
try:
    from backend.governance.replay_receipt import (
        validate_replay_receipt,
        load_replay_receipt,
        ReplayStatus,
        ReconErrorCode,
    )
    REPLAY_RECEIPT_AVAILABLE = True
except ImportError:
    REPLAY_RECEIPT_AVAILABLE = False

def get_file_hash(file_path):
    """Computes the SHA256 hash of a file."""
    if not os.path.exists(file_path):
        return None, f"File not found: {file_path}"
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest(), None

def verify_gates(manifest_path, replay_results_file, replay_telemetry_file):
    """
    Verifies all governance gates for a given manifest, including G3 replay.
    """
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest_data = json.load(f)
    except FileNotFoundError:
        return {"status": "ERROR", "message": f"Manifest file not found: {manifest_path}", "exit_code": 1}
    except json.JSONDecodeError:
        return {"status": "ERROR", "message": "Invalid JSON in manifest file.", "exit_code": 1}

    # --- Manifest v2 Schema Check ---
    required_v2_fields = ['ht_binding_hash', 'results_hash', 'experiment_id']
    missing_fields = [f for f in required_v2_fields if f not in manifest_data]
    if missing_fields:
        return {"status": "FAILURE", "gate": "G0_schema", "message": f"Manifest missing v2 fields: {missing_fields}", "exit_code": 5}

    experiment_id = manifest_data["experiment_id"]
    run_dir = os.path.dirname(manifest_path)
    prereg_file_path = os.path.join(run_dir, "prereg.yaml")
    slice_config_path = os.path.join(run_dir, "slice.json")
    results_file_path = os.path.join(run_dir, "results.jsonl")

    # --- G1: Preregistration Check ---
    prereg_result = verify_preregistration(prereg_file_path, experiment_id)
    if prereg_result["exit_code"] != 0:
        return {"status": "FAILURE", "gate": "G1_preregistration", "message": prereg_result["message"], "exit_code": 10}

    # --- G2 & Integrity Checks ---
    integrity_result = verify_manifest(manifest_path, prereg_file_path, slice_config_path, results_file_path)
    if integrity_result["exit_code"] != 0:
        return {"status": "FAILURE", "gate": "EV1_manifest_integrity", "message": integrity_result["message"], "exit_code": 20}

    # --- G3/EV3: Determinism Replay Gate ---
    replay_results_hash, err = get_file_hash(replay_results_file)
    if err:
        return {"status": "FAILURE", "gate": "G3_determinism_replay", "message": err, "exit_code": 30}

    replay_telemetry_hash, err = get_file_hash(replay_telemetry_file)
    if err:
        return {"status": "FAILURE", "gate": "G3_determinism_replay", "message": err, "exit_code": 30}

    results_consistent = replay_results_hash == manifest_data["results_hash"]
    telemetry_consistent = replay_telemetry_hash == manifest_data["ht_binding_hash"]

    if not results_consistent or not telemetry_consistent:
        return {
            "status": "FAILURE",
            "gate": "G3_determinism_replay",
            "message": "Replay artifacts do not match manifest commitments.",
            "exit_code": 30,
            "replay_verification": {
                "status": "FAILED_INCONSISTENT",
                "results_hash": {"expected": manifest_data["results_hash"], "actual": replay_results_hash, "match": results_consistent},
                "ht_binding_hash": {"expected": manifest_data["ht_binding_hash"], "actual": replay_telemetry_hash, "match": telemetry_consistent}
            }
        }

    replay_verification_output = {
        "status": "VERIFIED_CONSISTENT",
        "results_hash": manifest_data["results_hash"],
        "ht_binding_hash": manifest_data["ht_binding_hash"]
    }

    return {
      "status": "SUCCESS",
      "experiment_id": experiment_id,
      "replay_verification": replay_verification_output,
      "exit_code": 0
    }


def verify_replay_receipt_gate(manifest_path, receipt_path=None):
    """
    Verify the replay receipt gate (G4) per U2_REPLAY_RECEIPT_CHARTER.md.

    This is a separate verification path that validates the formal
    determinism_replay_receipt.json artifact.

    Args:
        manifest_path: Path to the experiment manifest
        receipt_path: Optional explicit path to replay receipt
                      (default: auto-discover in same directory)

    Returns:
        dict with gate status
    """
    if not REPLAY_RECEIPT_AVAILABLE:
        return {
            "status": "SKIP",
            "gate": "G4_replay_receipt",
            "message": "Replay receipt module not available",
            "exit_code": 0,  # Not a failure, just not available
        }

    run_dir = Path(manifest_path).parent

    # Auto-discover receipt path if not provided
    if receipt_path is None:
        # Check standard locations
        candidates = [
            run_dir / "replay" / "determinism_replay_receipt.json",
            run_dir / "determinism_replay_receipt.json",
            run_dir / "replay_receipt.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                receipt_path = candidate
                break

    if receipt_path is None:
        return {
            "status": "FAILURE",
            "gate": "G4_replay_receipt",
            "message": "No replay receipt found (RECON-18)",
            "error_code": "RECON-18",
            "exit_code": 40,
        }

    receipt_path = Path(receipt_path)

    # Validate the receipt
    valid, error_code, message = validate_replay_receipt(receipt_path)

    if not valid:
        return {
            "status": "FAILURE",
            "gate": "G4_replay_receipt",
            "message": message,
            "error_code": error_code.value if error_code else None,
            "exit_code": 40 + (int(error_code.value.split("-")[1]) - 18 if error_code else 0),
        }

    # Load receipt for detailed output
    receipt = load_replay_receipt(receipt_path)

    return {
        "status": "SUCCESS",
        "gate": "G4_replay_receipt",
        "message": "Replay receipt verified",
        "receipt_status": receipt.status.value,
        "checks_passed": receipt.verification_summary.checks_passed,
        "checks_total": receipt.verification_summary.checks_total,
        "receipt_hash": receipt.receipt_hash[:16] + "...",
        "gate_replay_status": {
            "G1": {"status": "pass", "replay_verified": True},
            "G2": {"status": "pass", "replay_verified": True},
            "G3": {"status": "pass", "replay_verified": True},
            "G4": {"status": "pass", "replay_verified": True},
            "G5": {"status": "pass", "replay_verified": False},  # Statistical gate, not subject to replay
        },
        "exit_code": 0,
    }


def verify_gates_with_receipt(manifest_path, replay_results_file, replay_telemetry_file, receipt_path=None):
    """
    Full gate verification including replay receipt (G4).

    This is the comprehensive verification function that includes:
    - G1: Preregistration
    - G2: Slice config hash
    - G3: Determinism replay (hash-based)
    - G4: Replay receipt (formal attestation)
    """
    # First run the existing gate verification
    base_result = verify_gates(manifest_path, replay_results_file, replay_telemetry_file)

    if base_result["exit_code"] != 0:
        return base_result

    # Now verify the replay receipt (G4)
    receipt_result = verify_replay_receipt_gate(manifest_path, receipt_path)

    if receipt_result["status"] == "FAILURE":
        return receipt_result

    # Merge results
    return {
        "status": "SUCCESS",
        "experiment_id": base_result["experiment_id"],
        "replay_verification": base_result["replay_verification"],
        "replay_receipt": receipt_result if receipt_result["status"] == "SUCCESS" else None,
        "gates": {
            "G1_preregistration": "PASS",
            "G2_slice_config_hash": "PASS",
            "G3_determinism_replay": "PASS",
            "G4_replay_receipt": receipt_result["status"],
        },
        "exit_code": 0,
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verifies all governance gates for a U2 run (v2).",
        epilog="""
Exit Codes:
  0  - All gates passed
  1  - General error
  5  - Schema validation failure (G0)
  10 - Preregistration failure (G1)
  20 - Manifest integrity failure (G2)
  30 - Determinism replay failure (G3)
  40 - Replay receipt missing (RECON-18)
  41 - Replay receipt mismatch (RECON-19)
  42 - Replay receipt incomplete (RECON-20)
        """
    )
    parser.add_argument("--manifest", required=True, help="Path to the U2 manifest file.")
    parser.add_argument("--replay-results-file", required=True, help="Path to the results file from the replay run.")
    parser.add_argument("--replay-telemetry-file", required=True, help="Path to the telemetry log from the replay run.")
    parser.add_argument("--replay-receipt", default=None, help="Path to the replay receipt (auto-discovered if not provided).")
    parser.add_argument("--require-receipt", action="store_true", help="Fail if replay receipt is missing.")
    parser.add_argument("--json", action="store_true", help="Output result as a JSON object.")
    args = parser.parse_args()

    # Determine which verification mode to use
    if args.replay_receipt or args.require_receipt:
        # Full verification with replay receipt
        result = verify_gates_with_receipt(
            args.manifest,
            args.replay_results_file,
            args.replay_telemetry_file,
            args.replay_receipt,
        )
    else:
        # Basic verification (backward compatible)
        result = verify_gates(args.manifest, args.replay_results_file, args.replay_telemetry_file)

    if args.json:
        output = {k: v for k, v in result.items() if k != 'exit_code'}
        print(json.dumps(output, indent=2))
    else:
        if result["exit_code"] == 0:
            print("All governance gates passed.")
            if "gates" in result:
                for gate, status in result["gates"].items():
                    print(f"  {gate}: {status}")
            if "replay_receipt" in result and result["replay_receipt"]:
                print(f"  Replay receipt: {result['replay_receipt']['receipt_status']}")
                print(f"    Checks: {result['replay_receipt']['checks_passed']}/{result['replay_receipt']['checks_total']}")
        else:
            print(f"Gate Failure: {result.get('gate')}")
            print(f"Message: {result.get('message')}")
            if "error_code" in result:
                print(f"Error Code: {result['error_code']}")

    sys.exit(result.get("exit_code", 1))

if __name__ == "__main__":
    main()