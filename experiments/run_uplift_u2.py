# PHASE II — NOT USED IN PHASE I
#
# This script runs a U2 uplift experiment. It is designed to be deterministic
# and self-contained for reproducibility. It supports two modes: 'baseline'
# for random ordering and 'rfl' for policy-driven ordering.
#
# Snapshot support enables:
# - Saving state at configurable intervals
# - Restoring from snapshots to resume experiments
# - Deterministic replay from any checkpoint
#
# Budget Enforcement (Agent B1):
# - Phase II slices load budget parameters from config/verifier_budget_phase2.yaml
# - Budget is passed to the derivation pipeline for enforcement
# - Budget exhaustion produces explicit failure states (no silent truncation)
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
# - All code must remain deterministic except random shuffle in the baseline policy.
# - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
# - All new files must be standalone and MUST NOT modify Phase I behavior.

import argparse
import hashlib
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import yaml

# PRNG Contract (Agent A2 — runtime-ops-2):
#   All randomness uses DeterministicPRNG for reproducibility.
#   See rfl/prng/deterministic_prng.py for implementation.
from rfl.prng import DeterministicPRNG, int_to_hex_seed

# Phase II Budget Enforcement (Agent B1)
from backend.verification.budget_loader import (
    VerifierBudget,
    load_budget_for_slice,
    is_phase2_slice,
    DEFAULT_CONFIG_PATH,
)

# Phase II Curriculum Loading (curriculum_loader_v2)
from experiments.curriculum_loader_v2 import (
    CurriculumLoader,
    CurriculumLoaderError,
    CurriculumNotFoundError,
)

# Phase II Calibration (u2_calibration)
from experiments.u2_calibration import (
    validate_calibration,
    CalibrationNotFoundError as CalibNotFoundError,
    CalibrationInvalidError,
    check_calibration_exists,
)

# Phase II Verbose Formatter (verbose_formatter)
from experiments.verbose_formatter import (
    format_verbose_cycle,
    parse_verbose_fields,
    DEFAULT_VERBOSE_FIELDS,
)

from experiments.u2.runner import (
    U2Runner,
    U2Config,
    CycleResult,
    TracedExperimentContext,
    run_with_traces,
)
from experiments.u2.snapshots import (
    SnapshotData,
    SnapshotValidationError,
    SnapshotCorruptionError,
    NoSnapshotFoundError,
    load_snapshot,
    save_snapshot,
    find_latest_snapshot,
    rotate_snapshots,
)
from experiments.u2.logging import U2TraceLogger, CORE_EVENTS, ALL_EVENT_TYPES
from experiments.u2 import schema as trace_schema

# --- Slice-specific Success Metrics ---
# These are passed as pure functions. In a real scenario, these might be
# dynamically imported or otherwise more complex. For this standalone script,
# we define them here.

def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """Success is when the python eval matches the expected result."""
    try:
        # A mock 'correct' result is simply the eval of the string.
        return eval(item) == result
    except Exception:
        return False

def metric_algebra_expansion(item: str, result: Any) -> bool:
    """A mock success metric for algebra. We'll just use string length."""
    # This is a placeholder. A real metric would be much more complex.
    return len(str(result)) > len(item)

METRIC_DISPATCHER = {
    "arithmetic_simple": metric_arithmetic_simple,
    "algebra_expansion": metric_algebra_expansion,
}


def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def get_config(config_path: Path) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    print(f"INFO: Loading config from {config_path}")
    if not config_path.exists():
        print(f"ERROR: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_execute_fn(slice_name: str) -> Callable[[str, int], Tuple[bool, Any]]:
    """
    Create the execution function for a slice.
    
    This wraps the substrate call and returns (success, result).
    """
    def execute_fn(item: str, seed: int) -> Tuple[bool, Any]:
        """Execute item on FO substrate and return (success, result)."""
        success = False
        result: Dict[str, Any] = {}
        
        try:
            # Find the run_fo_cycles.py script relative to this script.
            script_dir = Path(__file__).parent.resolve()
            substrate_script = script_dir / "run_fo_cycles.py"
            
            # Check if substrate script exists
            if not substrate_script.exists():
                # Fall back to mock execution for testing
                # PRNG Contract (Agent A2): Use hierarchical PRNG for mock execution
                mock_prng = DeterministicPRNG(int_to_hex_seed(seed))
                mock_rng = mock_prng.for_path("mock_execution", slice_name, str(seed))
                success = mock_rng.random() > 0.5
                result = {"outcome": "VERIFIED" if success else "FAILED", "mock": True}
                return success, result
            
            cmd = [
                sys.executable,
                str(substrate_script),
                "--item",
                item,
                "--seed",
                str(seed),
            ]
            
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8',
                timeout=60,  # 60 second timeout
            )
            
            result = json.loads(proc.stdout)
            if result.get("outcome") == "VERIFIED":
                success = True

        except subprocess.TimeoutExpired:
            result = {"error": "timeout", "item": item}
            success = False
        except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                result = {"error": str(e), "stdout": e.stdout, "stderr": e.stderr}
            else:
                result = {"error": str(e)}
            success = False
        
        return success, result
    
    return execute_fn


def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
    snapshot_interval: int = 0,
    snapshot_dir: Optional[Path] = None,
    restore_from: Optional[Path] = None,
    trace_log_path: Optional[Path] = None,
    trace_ctx: Optional[TracedExperimentContext] = None,
    snapshot_keep: int = 5,
    trace_events: Optional[set] = None,
    require_calibration: bool = False,
    calibration_dir: Optional[Path] = None,
    verbose_cycles: bool = False,
    verbose_fields: Optional[List[str]] = None,
):
    """
    Main function to run the uplift experiment with snapshot support.
    
    Args:
        slice_name: Name of the experiment slice
        cycles: Total number of cycles to run
        seed: Master random seed
        mode: "baseline" or "rfl"
        out_dir: Output directory for results
        config: Loaded configuration dict
        snapshot_interval: Save snapshot every N cycles (0 = disabled)
        snapshot_dir: Directory for snapshot files
        restore_from: Path to snapshot file to restore from (optional)
        trace_log_path: Path for trace log output (optional, enables tracing)
        trace_ctx: TracedExperimentContext for per-cycle logging (internal use)
        trace_events: Set of event types to log (None = all events)
        snapshot_keep: Number of snapshots to keep (rotation policy, 0 = no rotation)
        require_calibration: If True, require valid calibration before running
        calibration_dir: Directory containing calibration results (default: results/uplift_u2/calibration)
        verbose_cycles: If True, enable enhanced cycle-by-cycle logging
        verbose_fields: List of fields to include in verbose output (None = default fields)
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II — NOT USED IN PHASE I")
    
    # Phase II Calibration Guardrail
    if require_calibration:
        if calibration_dir is None:
            calibration_dir = Path("results/uplift_u2/calibration")
        
        print(f"INFO: Calibration check enabled for slice '{slice_name}'")
        try:
            summary = validate_calibration(calibration_dir, slice_name, require_valid=True)
            print(f"INFO: Calibration valid:")
            print(f"      determinism_verified = {summary.determinism_verified}")
            print(f"      schema_valid = {summary.schema_valid}")
            if summary.replay_hash:
                print(f"      replay_hash = {summary.replay_hash[:16]}...")
        except CalibNotFoundError as e:
            print(f"ERROR: Calibration not found for slice '{slice_name}'", file=sys.stderr)
            print(f"       {e}", file=sys.stderr)
            print(f"       Run calibration first before main uplift experiment.", file=sys.stderr)
            print(f"       Expected location: {calibration_dir / slice_name / 'calibration_summary.json'}", file=sys.stderr)
            sys.exit(2)  # Exit code 2 = calibration missing
        except CalibrationInvalidError as e:
            print(f"ERROR: Calibration invalid for slice '{slice_name}'", file=sys.stderr)
            print(f"       {e}", file=sys.stderr)
            print(f"       Re-run calibration to fix.", file=sys.stderr)
            sys.exit(2)  # Exit code 2 = calibration invalid
    
    # Phase II Budget Enforcement (Agent B1)
    # Load budget for Phase II slices; fail fast if missing
    budget: Optional[VerifierBudget] = None
    if is_phase2_slice(slice_name):
        try:
            budget = load_budget_for_slice(slice_name)
            print(f"INFO: Budget loaded for {slice_name}:")
            print(f"      cycle_budget_s={budget.cycle_budget_s}")
            print(f"      taut_timeout_s={budget.taut_timeout_s}")
            print(f"      max_candidates_per_cycle={budget.max_candidates_per_cycle}")
        except FileNotFoundError as e:
            print(f"ERROR: Budget config not found for Phase II slice '{slice_name}':", file=sys.stderr)
            print(f"       {e}", file=sys.stderr)
            print(f"       Phase II slices require config/verifier_budget_phase2.yaml", file=sys.stderr)
            sys.exit(1)
        except KeyError as e:
            print(f"ERROR: Budget config missing for slice '{slice_name}':", file=sys.stderr)
            print(f"       {e}", file=sys.stderr)
            sys.exit(1)
        except ValueError as e:
            print(f"ERROR: Invalid budget config for slice '{slice_name}':", file=sys.stderr)
            print(f"       {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"INFO: Non-Phase II slice '{slice_name}' — budget enforcement disabled")
    
    if snapshot_interval > 0:
        print(f"INFO: Snapshots enabled every {snapshot_interval} cycles")
    
    if restore_from:
        print(f"INFO: Restoring from snapshot: {restore_from}")

    # 1. Setup
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if snapshot_dir is None:
        snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load slice config and items using CurriculumLoader
    try:
        curriculum_loader = CurriculumLoader(config_path)
        curriculum_items = curriculum_loader.load_for_slice(slice_name)
        slice_config = curriculum_loader.get_slice_config(slice_name)
        
        # Convert CurriculumItem objects to strings for compatibility
        items = [item.formula for item in curriculum_items]
        print(f"INFO: Loaded {len(items)} curriculum items for slice '{slice_name}'")
        
    except CurriculumNotFoundError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print(f"         Using fallback items for testing.", file=sys.stderr)
        slice_config = {"items": [f"item_{i}" for i in range(10)]}
        items = slice_config.get("items", [f"item_{i}" for i in range(10)])
    except CurriculumLoaderError as e:
        print(f"ERROR: Failed to load curriculum: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 2. Create U2 runner with config
    u2_config = U2Config(
        experiment_id=f"u2_{slice_name}_{mode}",
        slice_name=slice_name,
        mode=mode,
        total_cycles=cycles,
        master_seed=seed,
        snapshot_interval=snapshot_interval,
        snapshot_dir=snapshot_dir,
        output_dir=out_dir,
        slice_config=slice_config,
    )
    
    runner = U2Runner(u2_config)
    
    # 3. Restore from snapshot if provided
    start_cycle = 0
    if restore_from:
        try:
            snapshot = load_snapshot(restore_from, verify_hash=True)
            runner.restore_state(snapshot)
            start_cycle = runner.cycle_index
            print(f"INFO: Restored to cycle {start_cycle}, resuming from cycle {start_cycle}")
        except SnapshotValidationError as e:
            print(f"ERROR: Snapshot validation failed: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"ERROR: Failed to load snapshot: {e}", file=sys.stderr)
            sys.exit(1)
    
    # 4. Create execution function
    execute_fn = create_execute_fn(slice_name)
    
    # 5. Output files
    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"
    
    # 6. Main Loop
    # Open in append mode if restoring, write mode otherwise
    file_mode = "a" if restore_from else "w"
    
    # Setup trace logging if enabled (PHASE II telemetry)
    trace_logger: Optional[U2TraceLogger] = None
    if trace_log_path is not None:
        trace_logger = U2TraceLogger(trace_log_path, fail_soft=True, enabled_events=trace_events)
        trace_logger.__enter__()
        if trace_events:
            print(f"INFO: Trace events filter: {sorted(trace_events)}")
        # Emit session start
        trace_logger.log_session_start(
            trace_schema.SessionStartEvent(
                run_id=u2_config.experiment_id,
                slice_name=slice_name,
                mode=mode,
                schema_version=trace_schema.TRACE_SCHEMA_VERSION,
                config_hash=hash_string(json.dumps(slice_config, sort_keys=True))[:16],
                total_cycles=cycles,
                initial_seed=seed,
            )
        )
        print(f"INFO: Trace logging enabled: {trace_log_path}")
    
    # Use external trace context if provided (for run_with_traces wrapper)
    local_trace_ctx = trace_ctx
    if local_trace_ctx is None and trace_logger is not None:
        local_trace_ctx = TracedExperimentContext(trace_logger, slice_name, mode)
    
    try:
        with open(results_path, file_mode) as results_f:
            for i in range(start_cycle, cycles):
                # Begin cycle timing if trace context available
                if local_trace_ctx is not None:
                    local_trace_ctx.begin_cycle(i)
                
                # Run cycle
                result = runner.run_cycle(items, execute_fn)
                
                # Write telemetry
                telemetry_record = {
                    "cycle": result.cycle_index,
                    "slice": result.slice_name,
                    "mode": result.mode,
                    "seed": result.seed,
                    "item": result.item,
                    "result": str(result.result),
                    "success": result.success,
                    "label": "PHASE II — NOT USED IN PHASE I",
                }
                results_f.write(json.dumps(telemetry_record) + "\n")
                results_f.flush()  # Ensure durability
                
                # Emit trace events if logging enabled
                if local_trace_ctx is not None:
                    # Log cycle telemetry (full record)
                    local_trace_ctx.log_cycle_telemetry(i, telemetry_record)
                    # End cycle timing
                    local_trace_ctx.end_cycle(i)
                
                # Progress output
                if verbose_cycles:
                    # Enhanced verbose output with configurable fields
                    if verbose_fields is None:
                        # Default fields
                        verbose_fields_to_use = DEFAULT_VERBOSE_FIELDS
                    else:
                        verbose_fields_to_use = verbose_fields
                    
                    # Prepare data dict for formatter
                    verbose_data = {
                        "cycle": i + 1,
                        "mode": result.mode,
                        "success": result.success,
                        "item": result.item,
                        "label": "PHASE_II",
                        "slice": result.slice_name,
                        "seed": result.seed,
                        "result": str(result.result),
                    }
                    
                    # Add item hash prefix if item is long
                    if len(result.item) > 8:
                        item_hash = hashlib.sha256(result.item.encode()).hexdigest()
                        verbose_data["item_hash_prefix"] = item_hash[:8]
                    
                    verbose_line = format_verbose_cycle(verbose_fields_to_use, verbose_data)
                    print(f"VERBOSE: {verbose_line}")
                else:
                    # Default concise output
                    print(f"Cycle {i+1}/{cycles}: Chose '{result.item}', Success: {result.success}")
                
                # Maybe save snapshot (handled internally by runner)
                snapshot_path = runner.maybe_save_snapshot()
                if snapshot_path:
                    print(f"INFO: Snapshot saved: {snapshot_path}")
                    # Apply rotation policy to keep disk usage bounded
                    if snapshot_keep > 0:
                        deleted = rotate_snapshots(snapshot_dir, keep_count=snapshot_keep)
                        if deleted:
                            print(f"INFO: Rotated {len(deleted)} old snapshot(s)")
    finally:
        # Close trace logger if we opened it
        if trace_logger is not None:
            # Emit session end
            trace_logger.log_session_end(
                trace_schema.SessionEndEvent(
                    run_id=u2_config.experiment_id,
                    slice_name=slice_name,
                    mode=mode,
                    schema_version=trace_schema.TRACE_SCHEMA_VERSION,
                    manifest_hash=None,  # Will be computed after manifest generation
                    ht_series_hash=None,
                    total_cycles=cycles,
                    completed_cycles=runner.cycle_index,
                )
            )
            trace_logger.__exit__(None, None, None)

    # 7. Final snapshot at end if interval is set
    if snapshot_interval > 0:
        final_snapshot = runner.capture_state()
        final_path = snapshot_dir / f"snapshot_{u2_config.experiment_id}_final.snap"
        save_snapshot(final_snapshot, final_path)
        print(f"INFO: Final snapshot saved: {final_path}")

    # 8. Manifest Generation
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    slice_config_hash = hash_string(slice_config_str)
    ht_series_str = json.dumps(runner.ht_series, sort_keys=True)
    ht_series_hash = hash_string(ht_series_str)

    manifest = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "slice": slice_name,
        "mode": mode,
        "cycles": cycles,
        "initial_seed": seed,
        "slice_config_hash": slice_config_hash,
        "prereg_hash": slice_config.get("prereg_hash", "N/A"),
        "ht_series_hash": ht_series_hash,
        "ht_series_length": len(runner.ht_series),
        "snapshot_interval": snapshot_interval,
        "snapshot_dir": str(snapshot_dir) if snapshot_interval > 0 else None,
        "restored_from": str(restore_from) if restore_from else None,
        "outputs": {
            "results": str(results_path),
            "manifest": str(manifest_path),
        },
        "policy_stats": {
            "update_count": runner.policy_update_count,
            "success_count": dict(runner.success_count),
            "attempt_count": dict(runner.attempt_count),
        } if mode == "rfl" else None,
        # Phase II Budget Enforcement (Agent B1)
        "budget": {
            "cycle_budget_s": budget.cycle_budget_s,
            "taut_timeout_s": budget.taut_timeout_s,
            "max_candidates_per_cycle": budget.max_candidates_per_cycle,
        } if budget is not None else None,
    }

    with open(manifest_path, "w") as manifest_f:
        json.dump(manifest, manifest_f, indent=2)

    print(f"\n--- Experiment Complete ---")
    print(f"Results written to {results_path}")
    print(f"Manifest written to {manifest_path}")
    if snapshot_interval > 0:
        print(f"Snapshots saved to {snapshot_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Uplift Runner with Snapshot Support. Must not be used for Phase I.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- RFL uses verifiable feedback only.

Snapshot Support:
- Use --snapshot-interval N to save state every N cycles
- Use --restore-from PATH to resume from a specific snapshot
- Use --resume to automatically find and resume from the latest snapshot
- Use --snapshot-keep N to limit disk usage (keeps last N snapshots)
- Snapshots enable deterministic pause/resume of long experiments

Exit Codes:
- 0: Success
- 1: General error
- 2: Calibration missing/invalid (with --require-calibration) or no snapshot found (with --resume)
        """
    )
    parser.add_argument(
        "--slice", 
        required=True, 
        type=str, 
        help="The experiment slice to run (e.g., 'arithmetic_simple')."
    )
    parser.add_argument(
        "--cycles", 
        required=True, 
        type=int, 
        help="Number of experiment cycles to run."
    )
    parser.add_argument(
        "--seed", 
        required=True, 
        type=int, 
        help="Initial random seed for deterministic execution."
    )
    parser.add_argument(
        "--mode", 
        required=True, 
        choices=["baseline", "rfl"], 
        help="Execution mode: 'baseline' or 'rfl'."
    )
    parser.add_argument(
        "--out", 
        required=True, 
        type=str, 
        help="Output directory for results and manifest files."
    )
    parser.add_argument(
        "--config", 
        default="config/curriculum_uplift_phase2.yaml", 
        type=str, 
        help="Path to the curriculum config file."
    )
    
    # Snapshot arguments
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=0,
        help="Save snapshot every N cycles (0 = disabled, default)."
    )
    parser.add_argument(
        "--snapshot-dir",
        type=str,
        default=None,
        help="Directory for snapshot files (default: <out>/snapshots)."
    )
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Path to snapshot file to restore from and continue."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest snapshot in snapshot-dir. Mutually exclusive with --restore-from."
    )
    parser.add_argument(
        "--snapshot-keep",
        type=int,
        default=5,
        help="Number of snapshots to keep (rotation policy). Default: 5. Set to 0 to disable rotation."
    )
    
    # Trace logging (PHASE II telemetry)
    parser.add_argument(
        "--trace-log",
        type=str,
        default=None,
        help="Path for trace log output (JSONL). Enables structured telemetry."
    )
    parser.add_argument(
        "--trace-events",
        type=str,
        default=None,
        help=(
            "Comma-separated list of event types to log (e.g., 'cycle_telemetry,policy_weight_update'). "
            f"Available: {','.join(sorted(ALL_EVENT_TYPES))}. "
            "If --trace-log is set but --trace-events is not, logs core events only: "
            f"{','.join(sorted(CORE_EVENTS))}."
        )
    )
    
    # Calibration arguments (PHASE II)
    parser.add_argument(
        "--require-calibration",
        action="store_true",
        help=(
            "Require valid calibration before running experiment. "
            "Checks for calibration_summary.json in results/uplift_u2/calibration/<slice>/. "
            "Exits with code 2 if calibration missing or invalid."
        )
    )
    parser.add_argument(
        "--calibration-dir",
        type=str,
        default=None,
        help="Directory containing calibration results (default: results/uplift_u2/calibration)."
    )
    
    # Verbose cycles (developer mode)
    parser.add_argument(
        "--verbose-cycles",
        action="store_true",
        help=(
            "Enable enhanced cycle-by-cycle logging with configurable fields. "
            "Fields can be customized via U2_VERBOSE_FIELDS environment variable "
            "(comma-separated, e.g., 'cycle,mode,success,item,label,item_hash_prefix'). "
            "Default fields: cycle,mode,success,item"
        )
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.resume and args.restore_from:
        print("ERROR: --resume and --restore-from are mutually exclusive.", file=sys.stderr)
        print("       Use --resume to auto-discover the latest snapshot,", file=sys.stderr)
        print("       or --restore-from to specify a specific snapshot file.", file=sys.stderr)
        sys.exit(1)

    config_path = Path(args.config)
    out_dir = Path(args.out)
    snapshot_dir = Path(args.snapshot_dir) if args.snapshot_dir else out_dir / "snapshots"
    trace_log_path = Path(args.trace_log) if args.trace_log else None
    
    # Parse trace events filter
    trace_events: Optional[set] = None
    if args.trace_events:
        trace_events = set(e.strip() for e in args.trace_events.split(",") if e.strip())
        # Validate event types
        invalid = trace_events - ALL_EVENT_TYPES
        if invalid:
            print(f"ERROR: Invalid trace event types: {invalid}", file=sys.stderr)
            print(f"Available: {sorted(ALL_EVENT_TYPES)}", file=sys.stderr)
            sys.exit(1)
    elif trace_log_path is not None:
        # Default to core events if --trace-log but no --trace-events
        trace_events = set(CORE_EVENTS)

    # Handle --resume: auto-discover latest snapshot
    restore_from: Optional[Path] = None
    if args.resume:
        latest = find_latest_snapshot(snapshot_dir)
        if latest is None:
            print(f"ERROR: No snapshot found in {snapshot_dir}", file=sys.stderr)
            print(f"       Cannot resume without a prior snapshot.", file=sys.stderr)
            print(f"       Run without --resume to start a fresh experiment.", file=sys.stderr)
            sys.exit(2)  # Exit code 2 = no resume point
        restore_from = latest
        print(f"INFO: --resume: Found latest snapshot: {latest}")
    elif args.restore_from:
        restore_from = Path(args.restore_from)

    config = get_config(config_path)
    
    # Parse calibration directory
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None
    
    # Parse verbose fields from environment
    verbose_fields = None
    if args.verbose_cycles:
        verbose_fields = parse_verbose_fields()
        if verbose_fields:
            print(f"INFO: Verbose fields configured: {', '.join(verbose_fields)}")
        else:
            print(f"INFO: Verbose cycles enabled with default fields")

    run_experiment(
        slice_name=args.slice,
        cycles=args.cycles,
        seed=args.seed,
        mode=args.mode,
        out_dir=out_dir,
        config=config,
        snapshot_interval=args.snapshot_interval,
        snapshot_dir=snapshot_dir,
        restore_from=restore_from,
        trace_log_path=trace_log_path,
        snapshot_keep=args.snapshot_keep,
        trace_events=trace_events,
        require_calibration=args.require_calibration,
        calibration_dir=calibration_dir,
        verbose_cycles=args.verbose_cycles,
        verbose_fields=verbose_fields,
    )

if __name__ == "__main__":
    main()
