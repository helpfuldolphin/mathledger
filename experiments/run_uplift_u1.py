#!/usr/bin/env python3
"""
RFL Uplift Experiment U1 Runner
================================

Preregistered experiment running baseline and RFL modes back-to-back
with slice_pl_uplift_a for measuring RFL uplift.

Usage:
    uv run python experiments/run_uplift_u1.py --cycles=300

Output:
    - results/uplift_u1_baseline.jsonl
    - results/uplift_u1_rfl.jsonl
    - results/uplift_u1_manifest.json
"""

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

print(f"[U1] __file__ = {__file__}", file=sys.stderr)

# Add project root to sys.path
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Top-level execution guard - this will print if the file is executed at all
try:
    with open("results/uplift_u1_top_level.txt", "w") as f:
        f.write("TOP-LEVEL EXECUTED\n")
        f.flush()
except Exception:
    pass
print("[U1] TOP-LEVEL EXECUTED", flush=True, file=sys.stderr)

try:
    from experiments.run_fo_cycles import CycleRunner
    print("[U1] CycleRunner imported successfully", flush=True, file=sys.stderr)
except Exception as e:
    print(f"[U1] ERROR importing CycleRunner: {e}", flush=True, file=sys.stderr)
    import traceback
    traceback.print_exc()
    raise

# Preregistered experiment parameters
UPLIFT_U1_PARAMS = {
    "experiment_id": "uplift_u1",
    "slice_name": "slice_pl_uplift_a",
    "system": "pl",
    "description": "RFL uplift measurement on slice_pl_uplift_a (atoms=4, depth_max=5)",
    "preregistered": True,
    "slice_params": {
        "atoms": 4,
        "depth_max": 5,
        "breadth_max": 800,
        "total_max": 4000,
    },
}


def run_uplift_u1(cycles: int = 300, output_dir: Path = Path("results/uplift_u1")) -> Dict[str, Any]:
    """
    Run Uplift Experiment U1: baseline and RFL modes with slice_pl_uplift_a.
    
    Args:
        cycles: Number of cycles to run for each mode
        output_dir: Directory to write output files (default: results/uplift_u1)
        
    Returns:
        Manifest dictionary with experiment metadata and file paths
    """
    # NEW: minimal logging to the same debug file used in main()
    try:
        dbg = Path("results/uplift_u1_debug.log")
        dbg.parent.mkdir(parents=True, exist_ok=True)
        with dbg.open("a") as f:
            f.write(f"[U1] Entered run_uplift_u1(cycles={cycles}, output_dir={output_dir})\n")
            f.flush()
    except Exception:
        # If even this fails, we ignore it – it's just debug.
        pass
    
    print(f"[U1] Starting run_uplift_u1: cycles={cycles}, output_dir={output_dir}", file=sys.stderr)
    
    # Explicitly create directory with debug print
    print(f"[U1] Creating output directory: {output_dir}", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[U1] Ensured directory exists: {output_dir} (exists={output_dir.exists()})", file=sys.stderr)
    
    baseline_path = output_dir / "baseline.jsonl"
    rfl_path = output_dir / "rfl.jsonl"
    manifest_path = output_dir / "experiment_manifest.json"
    
    print("=" * 60)
    print("RFL UPLIFT EXPERIMENT U1")
    print("=" * 60)
    print(f"Slice: {UPLIFT_U1_PARAMS['slice_name']}")
    print(f"Cycles per mode: {cycles}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Run baseline
    print("[U1] Starting baseline run...", file=sys.stderr)
    print("Running baseline mode...")
    try:
        baseline_runner = CycleRunner(
            mode="baseline",
            output_path=baseline_path,
            slice_name=UPLIFT_U1_PARAMS["slice_name"],
            system=UPLIFT_U1_PARAMS["system"],
        )
        print(f"[U1] Baseline runner created, calling run({cycles})...", file=sys.stderr)
        baseline_runner.run(cycles)
        print(f"[U1] Baseline complete → {baseline_path} (exists={baseline_path.exists()})", file=sys.stderr)
    except Exception as e:
        print(f"[U1] ERROR during baseline run: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
    
    # Run RFL
    print("[U1] Starting RFL run...", file=sys.stderr)
    print("\nRunning RFL mode...")
    try:
        rfl_runner = CycleRunner(
            mode="rfl",
            output_path=rfl_path,
            slice_name=UPLIFT_U1_PARAMS["slice_name"],
            system=UPLIFT_U1_PARAMS["system"],
        )
        print(f"[U1] RFL runner created, calling run({cycles})...", file=sys.stderr)
        rfl_runner.run(cycles)
        print(f"[U1] RFL complete → {rfl_path} (exists={rfl_path.exists()})", file=sys.stderr)
    except Exception as e:
        print(f"[U1] ERROR during RFL run: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
    
    # Compute preregistration hash if available
    prereg_path = Path("experiments/prereg/PREREG_UPLIFT_U1.md")
    prereg_hash = None
    prereg_hash_computed_at = None
    if prereg_path.exists():
        import hashlib
        prereg_content = prereg_path.read_bytes()
        prereg_hash = hashlib.sha256(prereg_content).hexdigest()
        prereg_hash_computed_at = datetime.now(timezone.utc).isoformat()
    
    # Get environment info
    import platform
    import subprocess
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    platform_str = platform.platform()
    mathledger_commit = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            mathledger_commit = result.stdout.strip()
    except Exception:
        pass
    
    # Create manifest matching preregistration schema
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "manifest_version": "1.0",
        "experiment_id": "uplift_u1",
        "preregistration": {
            "prereg_path": str(prereg_path.relative_to(Path.cwd())) if prereg_path.exists() else None,
            "prereg_hash_sha256": prereg_hash,
            "prereg_hash_computed_at": prereg_hash_computed_at,
        },
        "execution": {
            "started_at": started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "executor": "run_uplift_u1.py",
            "environment": {
                "python_version": python_version,
                "platform": platform_str,
                "mathledger_commit": mathledger_commit,
            },
        },
        "artifacts": {
            "baseline_log": str(baseline_path.relative_to(Path.cwd())),
            "rfl_log": str(rfl_path.relative_to(Path.cwd())),
            "baseline_attestation": None,  # To be filled by analyzer
            "rfl_attestation": None,  # To be filled by analyzer
            "statistical_summary": None,  # To be filled by analyzer
        },
        "determinism_verification": {
            "baseline_H_t": None,  # To be filled by analyzer
            "baseline_replay_H_t": None,  # To be filled by analyzer
            "baseline_determinism_match": None,  # To be filled by analyzer
            "rfl_H_t": None,  # To be filled by analyzer
            "rfl_replay_H_t": None,  # To be filled by analyzer
            "rfl_determinism_match": None,  # To be filled by analyzer
        },
        "outcome": None,  # To be filled by analyzer
        "gate_alignment": {
            "satisfies_vsd_phase_2_gate": None,  # To be filled by analyzer
            "gate_document": "docs/VSD_PHASE_2.md",
            "gate_section": "Phase II Uplift Evidence Gate",
        },
    }
    
    # Write manifest
    print(f"[U1] Writing experiment manifest → {manifest_path}", file=sys.stderr)
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"[U1] Manifest written (exists={manifest_path.exists()})", file=sys.stderr)
    except Exception as e:
        print(f"[U1] ERROR writing manifest: {e}", file=sys.stderr)
        traceback.print_exc()
        raise
    
    print(f"\n✅ Experiment complete!")
    print(f"   Baseline: {baseline_path}")
    print(f"   RFL:      {rfl_path}")
    print(f"   Manifest: {manifest_path}")
    print()
    print("Next step: Run analysis:")
    print(f"   uv run python experiments/analyze_uplift_u1.py")
    print("[U1] DONE.", file=sys.stderr)
    
    return manifest


def main():
    # Write to a log file immediately to debug output issues
    log_path = Path("results/uplift_u1_debug.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as log_file:
        log_file.write("[U1] Script started\n")
        log_file.flush()
        
        parser = argparse.ArgumentParser(
            description="Run RFL Uplift Experiment U1 (preregistered)"
        )
        parser.add_argument(
            "--cycles",
            type=int,
            default=300,
            help="Number of cycles per mode (default: 300)",
        )
        parser.add_argument(
            "--out-dir",
            type=str,
            default="results/uplift_u1",
            help="Output directory (default: results/uplift_u1)",
        )
        
        args = parser.parse_args()
        log_file.write(f"[U1] Requested cycles={args.cycles}, out_dir={args.out_dir}\n")
        log_file.flush()
        print(f"[U1] Requested cycles={args.cycles}, out_dir={args.out_dir}", file=sys.stderr)
        
        log_file.write("[U1] Reached try block\n")
        log_file.flush()
        
        try:
            log_file.write("[U1] About to call run_uplift_u1(...)\n")
            log_file.flush()
            print("[U1] About to call run_uplift_u1(...)", file=sys.stderr)
            
            manifest = run_uplift_u1(cycles=args.cycles, output_dir=Path(args.out_dir))
            
            log_file.write("[U1] Returned from run_uplift_u1(...)\n")
            log_file.write("[U1] main() completed successfully\n")
            log_file.flush()
            print("[U1] main() completed successfully", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            error_msg = f"[U1] ERROR: Experiment failed: {e}\n"
            log_file.write(error_msg)
            log_file.write(traceback.format_exc())
            log_file.flush()
            print(error_msg, file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    # Write to file immediately to prove execution
    try:
        with open("results/uplift_u1_execution_proof.txt", "w") as f:
            f.write(f"[U1] Script entry point reached\n")
            f.write(f"argv: {sys.argv}\n")
            f.flush()
    except Exception:
        pass
    print("[U1] Script entry point reached", file=sys.stderr)
    main()

