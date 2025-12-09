# scripts/check_determinism_over_history.py
"""
GEMINI-K: SIGMA-III - Cross-Run Determinism Auditor

This script provides an empirical guarantee of determinism by executing
the RFL runner multiple times with a fixed manifest and configuration.
It then inspects the results of each run to detect any divergence,
which would indicate a breach of determinism.

Invariant: For a fixed manifest, N executions must produce N identical
           and verifiable result hashes.
"""

import argparse
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

# --- Configuration ---

# The JSON path to the definitive, stable hash within the runner's output results file.
# This must be a hash that represents the final state of a successful run.
# The `composite_root` of the last ledger entry is a good candidate.
RESULT_HASH_JSON_PATH = ("policy", "ledger", -1, "composite_root")

# --- Audit Logic ---

def run_single_experiment(manifest_path: str, run_index: int) -> Optional[str]:
    """
    Executes a single RFL runner experiment as an isolated subprocess.

    Args:
        manifest_path: Path to the experiment config/manifest file.
        run_index: The index of the current run for logging.

    Returns:
        The result hash of the run if successful, otherwise None.
    """
    print(f"--- Starting Audit Run {run_index}... ---")
    
    # Define a unique output directory for this run's artifacts to prevent collisions.
    output_dir = Path(f"tmp/audit_run_{run_index}")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "results.json"
    
    command = [
        sys.executable,
        "rfl/runner.py",
        "--config",
        manifest_path,
    ]
    
    # We must override the default artifacts_dir in the config to ensure isolation.
    # A robust way is to pass an override. For this implementation, we assume
    # the runner's config can be modified or it outputs to a predictable location
    # that we can clean up. For now, we will assume the runner respects a
    # temporary artifact location passed via an environment variable if possible,
    # or we will parse the default output path from its config.
    
    # NOTE: This is a simplified integration. A production version would need a
    # more robust way to override the output path of the runner.
    # For now, we will assume the runner has been modified to accept a
    # temporary output path or we are cleaning up its default location.
    
    env = os.environ.copy()
    env["RFL_ENV_MODE"] = "PHASE-II-U2" # Enforce U2 security mode
    env["RFL_ARTIFACTS_DIR"] = str(output_dir) # Hypothetical override

    try:
        process = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=True, # Throws an exception for non-zero exit codes
        )
        print(f"Run {run_index} completed successfully.")
        
        # The runner must be modified to read config overrides for artifact paths,
        # otherwise this check will be flawed. Assuming it does:
        if not results_file.exists():
            print(f"ERROR: Results file '{results_file}' not found after run {run_index}.", file=sys.stderr)
            print("STDOUT:", process.stdout, file=sys.stderr)
            print("STDERR:", process.stderr, file=sys.stderr)
            return None

        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        # Traverse the JSON to find the result hash
        hash_value = results_data
        for key in RESULT_HASH_JSON_PATH:
            if isinstance(key, int) and isinstance(hash_value, list) and len(hash_value) > key:
                hash_value = hash_value[key]
            elif isinstance(key, str) and isinstance(hash_value, dict) and key in hash_value:
                hash_value = hash_value[key]
            else:
                print(f"ERROR: Could not find key '{key}' in results JSON for run {run_index}.", file=sys.stderr)
                return None
        
        if isinstance(hash_value, str) and len(hash_value) > 0:
            print(f"Run {run_index} result hash: {hash_value}")
            return hash_value
        else:
            print(f"ERROR: Extracted result hash is invalid for run {run_index}. Found: {hash_value}", file=sys.stderr)
            return None

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Audit Run {run_index} failed with exit code {e.returncode}.", file=sys.stderr)
        print("STDOUT:", e.stdout, file=sys.stderr)
        print("STDERR:", e.stderr, file=sys.stderr)
        return None
    except Exception as e:
        print(f"An unexpected error occurred during run {run_index}: {e}", file=sys.stderr)
        return None


def main():
    """Main entry point for the cross-run determinism auditor."""
    parser = argparse.ArgumentParser(description="GEMINI-K Cross-Run Determinism Auditor")
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to the fixed experiment manifest/config file to use for all runs."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="The number of times to execute the experiment."
    )
    args = parser.parse_args()

    if args.runs < 2:
        print("Error: Must perform at least 2 runs to check for divergence.", file=sys.stderr)
        sys.exit(1)

    print("--- GEMINI-K: Starting Cross-Run Determinism Audit ---")
    print(f"Manifest: {args.manifest}")
    print(f"Runs: {args.runs}")
    print("-" * 50)

    result_hashes: List[str] = []
    
    # This is a placeholder for a more robust artifact cleanup/isolation strategy
    import shutil
    if Path("tmp/audit_run_1").exists():
        print("Cleaning up previous audit artifacts...")
        shutil.rmtree("tmp")


    for i in range(1, args.runs + 1):
        result_hash = run_single_experiment(args.manifest, i)
        if result_hash is None:
            print("\nAudit Verdict: FAILED_DURING_RUN. An error occurred, preventing completion.", file=sys.stderr)
            sys.exit(1)
        result_hashes.append(result_hash)

    print("-" * 50)
    print("Audit Complete. Analyzing results...")
    
    first_hash = result_hashes[0]
    is_deterministic = all(h == first_hash for h in result_hashes)
    
    print("\n--- AUDIT SUMMARY ---")
    for i, h in enumerate(result_hashes):
        status = "MATCH" if h == first_hash else "!! DIVERGENCE !!"
        print(f"Run {i+1}: {h}  [{status}]")

    print("-" * 21)
    if is_deterministic:
        print("Audit Verdict: DETERMINISTIC")
        print("Conclusion: All runs produced identical result hashes.")
        sys.exit(0)
    else:
        print("Audit Verdict: NONDETERMINISTIC", file=sys.stderr)
        print("Conclusion: Divergence detected in result hashes across runs.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import os
    main()
