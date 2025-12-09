"""
PHASE II — NOT USED IN PHASE I
CI Gate Simulation Script

This script simulates the CI gates defined in the Metric Integration Spec.
It runs the stress tests, the auditor, and verification checks.
"""
import os
import subprocess
import json
import hashlib
import sys

def run_stage(command, stage_name):
    """Helper function to run a command and print its status."""
    print(f"--- Running Stage: {stage_name} ---")
    try:
        # Using shell=True for cross-platform compatibility with commands like 'python -m'
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        print(f"Stdout:\n{result.stdout}")
        print(f"Stderr:\n{result.stderr}")
        print(f"--- {stage_name}: PASSED ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- {stage_name}: FAILED ---")
        print(f"Exit Code: {e.returncode}")
        print(f"Stdout:\n{e.stdout}")
        print(f"Stderr:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"--- {stage_name}: FAILED ---")
        print(f"Error: The command '{command.split()[0]}' was not found.")
        print("Please ensure Python and its dependencies are installed and in the system's PATH.")
        return False


def main():
    print("====== Starting CI Gates for Metric Integration Correctness ======")
    
    # Stage 1: Original Stress Tests
    if not run_stage("python -m unittest tests/metrics/test_metric_stress.py", "Stage 1: Stress Tests"):
        print("\nCI checks failed at Stage 1. Aborting.")
        return

    # Stage 1b: New Test Suites
    if not run_stage("python -m unittest tests/metrics/test_streaming_auditor.py", "Stage 1b: Streaming Auditor Tests"):
        print("\nCI checks failed at Stage 1b. Aborting.")
        return
    if not run_stage("python -m unittest tests/metrics/test_ledger_store.py", "Stage 1b: Ledger Store Tests"):
        print("\nCI checks failed at Stage 1b. Aborting.")
        return
    if not run_stage("python -m unittest tests/metrics/test_phase3_4_audits.py", "Stage 1b: Phase 3/4 Audit Tests"):
        print("\nCI checks failed at Stage 1b. Aborting.")
        return
    if not run_stage("python -m unittest tests/metrics/test_metric_ledger.py", "Stage 1b: Ledger Property Tests"):
        print("\nCI checks failed at Stage 1b. Aborting.")
        return

    # Stage 2: Consistency Auditor (End-to-end check)
    if not run_stage("python experiments/metric_consistency_auditor.py", "Stage 2: Consistency Auditor"):
        print("\nCI checks failed at Stage 2. Aborting.")
        return

    # Stage 3 & 4: Ledger Verification and Determinism Re-run
    # In a real CI system, these would be separate stages.
    # For this simulation, we'll combine them into one step.
    print("\n--- Running Stage: 3 & 4 (Ledger Verification and Determinism) ---")
    try:
        # Generate the first ledger
        run_stage("python experiments/metric_consistency_auditor.py", "Ledger Generation (Run 1)")
        with open("artifacts/metrics/metric_integration_ledger.json", "r") as f1:
            ledger1 = json.load(f1)
            id1 = ledger1.get("ledger_id")

        # Generate the second ledger
        run_stage("python experiments/metric_consistency_auditor.py", "Ledger Generation (Run 2)")
        with open("artifacts/metrics/metric_integration_ledger.json", "r") as f2:
            ledger2 = json.load(f2)
            id2 = ledger2.get("ledger_id")

        # Compare ledger IDs for determinism
        if id1 == id2:
            print(f"Ledger IDs match: {id1}. Determinism check passed.")
        else:
            print(f"--- DETERMINISM FAILURE ---")
            print(f"Ledger IDs are not deterministic! ID1: {id1}, ID2: {id2}")
            sys.exit(1)
        
        # Verify the ledger ID (self-hash) by recalculating it
        ledger_body = {k: v for k, v in ledger1.items() if k not in ["ledger_id", "timestamp_utc"]}
        from experiments.metric_integration_ledger import to_canonical_json, hash_sha256
        recalculated_id = hash_sha256(to_canonical_json(ledger_body))
        
        assert id1 == recalculated_id, f"Ledger ID self-hash is incorrect! Expected {id1}, got {recalculated_id}"
        print("Ledger ID self-hash check passed.")

        print("--- Stages 3 & 4: PASSED ---")

    except (AssertionError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"--- Stages 3 & 4: FAILED ---")
        print(f"Error: {e}")
        return

    print("====== All CI gates passed successfully. ======")

if __name__ == "__main__":
    main()
