# PHASE II — U2 UPLIFT EXPERIMENT
# backend/runner/u2_runner.py
# Example U2 Experiment Runner demonstrating schema validation.

import uuid
import datetime
import random

# Assuming the telemetry module is in the path.
# In a real app, this might be handled by a proper package structure.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from telemetry.u2_schema import validate_cycle_event, validate_experiment_summary

def get_current_utc_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"

def generate_fake_hashes():
    """Generates fake 64-char hex hashes for H_t, R_t, U_t."""
    return {
        "H_t": uuid.uuid4().hex + uuid.uuid4().hex,
        "R_t": uuid.uuid4().hex + uuid.uuid4().hex,
        "U_t": uuid.uuid4().hex + uuid.uuid4().hex,
    }

def run_u2_arm(mode: str, slice_id: str, num_cycles: int) -> None:
    """
    Simulates running one arm (baseline or rfl) of a U2 experiment.
    """
    print(f"\n--- Running U2 Arm: mode='{mode}', slice='{slice_id}' ---")
    run_id = str(uuid.uuid4())
    
    for i in range(num_cycles):
        # 1. Generate a fake per-cycle event
        cycle_event = {
            "ts": get_current_utc_iso(),
            "run_id": run_id,
            "slice": slice_id,
            "mode": mode,
            "cycle": i,
            "success": random.random() > 0.15, # 85% success rate
            "metric_type": "duration_seconds",
            "metric_value": 1.0 + random.uniform(-0.2, 0.2),
            **generate_fake_hashes()
        }

        # 2. Validate the event before logging/sending it
        try:
            validate_cycle_event(cycle_event)
            if i % 50 == 0: # Log progress occasionally
                 print(f"Cycle {i}: Event validated successfully.")
        except (ValueError, TypeError) as e:
            print(f"[ERROR] Cycle {i}: Invalid cycle event data: {e}")
            # In a real scenario, this might halt the run or log a critical error
            return

    print(f"--- Finished U2 Arm: {mode} ---")
    return run_id


def generate_and_validate_summary(baseline_run_id: str, rfl_run_id: str):
    """
    Simulates generating and validating the final experiment summary manifest.
    """
    print("\n--- Generating and Validating Experiment Summary ---")
    
    # 1. Generate a fake summary
    summary = {
        "slice": "U2_env_A",
        "mode": "rfl_v2_exp1",
        "n_cycles": {
            "baseline": 100,
            "rfl": 100
        },
        "p_base": 0.85,
        "p_rfl": 0.88,
        "delta": 0.03,
        "CI": {
            "lower_bound": -0.05,
            "upper_bound": 0.11,
            "level": 0.95
        },
        "baseline_run_id": baseline_run_id,
        "rfl_run_id": rfl_run_id
    }

    # 2. Validate the summary before writing the manifest
    try:
        validate_experiment_summary(summary)
        print("Experiment summary manifest validated successfully.")
        # (Here you would write the summary to a manifest.json file)
    except (ValueError, TypeError) as e:
        print(f"[ERROR] Invalid experiment summary data: {e}")

if __name__ == "__main__":
    print("Starting U2 Telemetry Validation Runner...")
    
    # --- Simulate running a paired experiment ---
    baseline_id = run_u2_arm(mode="baseline", slice_id="U2_env_A", num_cycles=100)
    rfl_id = run_u2_arm(mode="rfl", slice_id="U2_env_A", num_cycles=100)

    # --- Simulate generating the final manifest ---
    if baseline_id and rfl_id:
        generate_and_validate_summary(baseline_id, rfl_id)
    
    print("\nU2 Runner finished.")
