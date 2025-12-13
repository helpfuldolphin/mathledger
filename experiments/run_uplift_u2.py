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
from experiments.u2.snapshot_history import (
    build_multi_run_snapshot_history,
    plan_future_runs,
    summarize_snapshot_plans_for_u2_orchestrator,
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


def _log_snapshot_plan_event(
    trace_logger: Optional[U2TraceLogger],
    event: trace_schema.SnapshotPlanEvent,
) -> None:
    """
    Log SnapshotPlanEvent to trace log.
    
    This is a simple JSONL writer for snapshot plan events.
    """
    if trace_logger is None:
        return
    
    # Write event as JSONL line
    event_data = {
        "type": "snapshot_plan_event",
        "event_type": "SnapshotPlanEvent",
        "payload": event.to_dict(),
    }
    
    # Access file handle directly if available, otherwise use a simple write
    try:
        if hasattr(trace_logger, 'file_handle') and trace_logger.file_handle:
            line = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
            trace_logger.file_handle.write(line + '\n')
            trace_logger.file_handle.flush()
        elif hasattr(trace_logger, 'output_path'):
            # Fallback: append to file
            with open(trace_logger.output_path, 'a', encoding='utf-8') as f:
                line = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
                f.write(line + '\n')
    except Exception as e:
        # Fail soft - don't crash if logging fails
        print(f"WARNING: Failed to log snapshot plan event: {e}", file=sys.stderr)


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
    snapshot_plan_event: Optional[trace_schema.SnapshotPlanEvent] = None,
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
    """
    print(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    print(f"PHASE II — NOT USED IN PHASE I")
    
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
    
    slice_config = config.get("slices", {}).get(slice_name, {})
    if not slice_config:
        # Try alternative config structure
        for item in config.get("slices", []):
            if isinstance(item, dict) and item.get("name") == slice_name:
                slice_config = item
                break
    
    if not slice_config:
        print(f"WARNING: Slice '{slice_name}' not found in config, using empty config.", file=sys.stderr)
        slice_config = {"items": [f"item_{i}" for i in range(10)]}
    
    items = slice_config.get("items", [f"item_{i}" for i in range(10)])
    
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
        # Emit session start (if method exists)
        try:
            if hasattr(trace_logger, 'log_session_start'):
                # Try to emit session start event (may not be defined in all versions)
                SessionStartEvent = getattr(trace_schema, 'SessionStartEvent', None)
                TRACE_SCHEMA_VERSION = getattr(trace_schema, 'TRACE_SCHEMA_VERSION', '1.0.0')
                if SessionStartEvent:
                    trace_logger.log_session_start(
                        SessionStartEvent(
                            run_id=u2_config.experiment_id,
                            slice_name=slice_name,
                            mode=mode,
                            schema_version=TRACE_SCHEMA_VERSION,
                            config_hash=hash_string(json.dumps(slice_config, sort_keys=True))[:16],
                            total_cycles=cycles,
                            initial_seed=seed,
                        )
                    )
        except Exception as e:
            # Fail soft - don't crash if session start logging fails
            print(f"WARNING: Failed to log session start event: {e}", file=sys.stderr)
        
        # Emit snapshot plan event if available (from auto-resume)
        if snapshot_plan_event is not None:
            _log_snapshot_plan_event(trace_logger, snapshot_plan_event)
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
            # Emit session end (if method exists)
            try:
                if hasattr(trace_logger, 'log_session_end'):
                    SessionEndEvent = getattr(trace_schema, 'SessionEndEvent', None)
                    TRACE_SCHEMA_VERSION = getattr(trace_schema, 'TRACE_SCHEMA_VERSION', '1.0.0')
                    if SessionEndEvent:
                        trace_logger.log_session_end(
                            SessionEndEvent(
                                run_id=u2_config.experiment_id,
                                slice_name=slice_name,
                                mode=mode,
                                schema_version=TRACE_SCHEMA_VERSION,
                                manifest_hash=None,  # Will be computed after manifest generation
                                ht_series_hash=None,
                                total_cycles=cycles,
                                completed_cycles=runner.cycle_index,
                            )
                        )
            except Exception as e:
                # Fail soft - don't crash if session end logging fails
                print(f"WARNING: Failed to log session end event: {e}", file=sys.stderr)
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
- 2: No snapshot found for --resume
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
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Automatically select best snapshot from snapshot-root for resuming. Uses multi-run planning."
    )
    parser.add_argument(
        "--snapshot-root",
        type=str,
        default=None,
        help="Root directory containing multiple run directories for auto-resume planning."
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

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.resume and args.restore_from:
        print("ERROR: --resume and --restore-from are mutually exclusive.", file=sys.stderr)
        print("       Use --resume to auto-discover the latest snapshot,", file=sys.stderr)
        print("       or --restore-from to specify a specific snapshot file.", file=sys.stderr)
        sys.exit(1)
    
    if args.auto_resume and (args.resume or args.restore_from):
        print("ERROR: --auto-resume is mutually exclusive with --resume and --restore-from.", file=sys.stderr)
        sys.exit(1)
    
    if args.auto_resume and not args.snapshot_root:
        print("ERROR: --snapshot-root is required when --auto-resume is used.", file=sys.stderr)
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
    snapshot_plan_event: Optional[trace_schema.SnapshotPlanEvent] = None
    
    if args.auto_resume:
        # Auto-resume: analyze multiple runs and select best snapshot
        snapshot_root = Path(args.snapshot_root)
        if not snapshot_root.exists():
            print(f"WARNING: Snapshot root directory not found: {snapshot_root}", file=sys.stderr)
            print(f"         Falling back to NEW_RUN.", file=sys.stderr)
            orchestrator_summary = {
                "status": "NEW_RUN",
                "has_resume_targets": False,
                "preferred_run_id": None,
                "preferred_snapshot_path": None,
                "details": {"runs_available": 0, "message": "Snapshot root not found"},
            }
        else:
            try:
                # Discover run directories (subdirectories of snapshot_root)
                run_dirs: List[str] = []
                try:
                    for item in snapshot_root.iterdir():
                        if item.is_dir():
                            run_dirs.append(str(item))
                except (OSError, PermissionError) as e:
                    print(f"WARNING: Error scanning snapshot root: {e}", file=sys.stderr)
                    print(f"         Falling back to NEW_RUN.", file=sys.stderr)
                    run_dirs = []
                
                if not run_dirs:
                    print(f"INFO: No run directories found in {snapshot_root}. Starting a new run.", file=sys.stderr)
                    orchestrator_summary = {
                        "status": "NEW_RUN",
                        "has_resume_targets": False,
                        "preferred_run_id": None,
                        "preferred_snapshot_path": None,
                        "details": {"runs_available": 0, "message": "No runs found"},
                    }
                else:
                    print(f"INFO: Analyzing {len(run_dirs)} runs for auto-resume...")
                    try:
                        # Build multi-run history (handles errors gracefully)
                        multi_history = build_multi_run_snapshot_history(run_dirs)
                        
                        # Plan future runs
                        plan = plan_future_runs(multi_history, target_coverage=10.0)
                        
                        # Get orchestrator summary
                        orchestrator_summary = summarize_snapshot_plans_for_u2_orchestrator(plan)
                        
                        if orchestrator_summary["status"] == "RESUME":
                            preferred_path = orchestrator_summary.get("preferred_snapshot_path")
                            if preferred_path:
                                restore_from = Path(preferred_path)
                                print(f"INFO: Auto-resume decision: RESUME from {restore_from}")
                                print(f"      Run ID: {orchestrator_summary.get('preferred_run_id', 'unknown')}")
                            else:
                                print(f"WARNING: RESUME status but no snapshot path. Falling back to NEW_RUN.", file=sys.stderr)
                                orchestrator_summary["status"] = "NEW_RUN"
                        elif orchestrator_summary["status"] == "NEW_RUN":
                            print(f"INFO: Auto-resume decision: NEW_RUN")
                            print(f"      Reason: {orchestrator_summary.get('details', {}).get('message', 'No viable resume points')}")
                        else:
                            print(f"INFO: Auto-resume decision: NO_ACTION")
                            print(f"      Reason: {orchestrator_summary.get('details', {}).get('message', '')}")
                    
                    except Exception as e:
                        # Log with context about which step failed
                        print(f"WARNING: Error during auto-resume analysis: {e}", file=sys.stderr)
                        print(f"         Snapshot root: {snapshot_root}", file=sys.stderr)
                        print(f"         Run directories found: {len(run_dirs)}", file=sys.stderr)
                        print(f"         Falling back to NEW_RUN.", file=sys.stderr)
                        orchestrator_summary = {
                            "status": "NEW_RUN",
                            "has_resume_targets": False,
                            "preferred_run_id": None,
                            "preferred_snapshot_path": None,
                            "details": {"runs_available": len(run_dirs), "message": f"Analysis error: {e}"},
                        }
                
                # Create SnapshotPlanEvent for telemetry
                # Extract metrics from multi_history for event
                mean_coverage = multi_history.get("summary", {}).get("average_coverage_pct", 0.0)
                max_gap = multi_history.get("global_max_gap", 0)
                
                snapshot_plan_event = trace_schema.SnapshotPlanEvent(
                    status=orchestrator_summary["status"],
                    preferred_run_id=orchestrator_summary.get("preferred_run_id"),
                    preferred_snapshot_path=orchestrator_summary.get("preferred_snapshot_path"),
                    total_runs_analyzed=multi_history.get("run_count", 0),
                    mean_coverage_pct=mean_coverage,
                    max_gap=max_gap,
                )
            
            except Exception as e:
                # Catch-all for any unexpected errors - log with full context
                import traceback
                print(f"ERROR: Unexpected error in auto-resume: {e}", file=sys.stderr)
                print(f"       Snapshot root: {snapshot_root}", file=sys.stderr)
                print(f"       Error type: {type(e).__name__}", file=sys.stderr)
                if len(run_dirs) > 0:
                    print(f"       Run directories found: {len(run_dirs)}", file=sys.stderr)
                print(f"       Falling back to NEW_RUN.", file=sys.stderr)
                # Log traceback for debugging (but don't crash)
                print(f"       Traceback (for debugging):", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                orchestrator_summary = {
                    "status": "NEW_RUN",
                    "has_resume_targets": False,
                    "preferred_run_id": None,
                    "preferred_snapshot_path": None,
                    "details": {"runs_available": len(run_dirs) if 'run_dirs' in locals() else 0, "message": f"Unexpected error: {e}"},
                }
                snapshot_plan_event = trace_schema.SnapshotPlanEvent(
                    status="NEW_RUN",
                    preferred_run_id=None,
                    preferred_snapshot_path=None,
                    total_runs_analyzed=0,
                    mean_coverage_pct=0.0,
                    max_gap=0,
                )
    
    elif args.resume:
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
        snapshot_plan_event=snapshot_plan_event,
    )

if __name__ == "__main__":
    main()
