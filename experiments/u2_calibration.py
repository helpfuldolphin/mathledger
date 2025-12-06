# PHASE II — NOT USED IN PHASE I
#
# Calibration Fire Harness for U2 Uplift Experiments
# ===================================================
#
# This module provides a "calibration mode" for U2 experiments:
# - Forces cycles=10 (small, cheap sanity runs)
# - Runs both baseline and RFL back-to-back
# - Writes logs to results/uplift_u2/calibration/<slice>/...
# - Checks determinism (replay works), schema validation, and success metrics
# - Emits a JSON summary per slice with diagnostics
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
# - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
# - No uplift statistics computed in calibration mode.

import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Default calibration cycles (small for cheap sanity runs)
CALIBRATION_CYCLES = 10


def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def validate_schema(record: Dict[str, Any]) -> List[str]:
    """
    Validate a telemetry record against the expected schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []
    required_fields = ["cycle", "slice", "mode", "seed", "item", "result", "success", "label"]
    for field in required_fields:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    if "cycle" in record and not isinstance(record["cycle"], int):
        errors.append(f"Field 'cycle' must be int, got {type(record['cycle']).__name__}")
    if "success" in record and not isinstance(record["success"], bool):
        errors.append(f"Field 'success' must be bool, got {type(record['success']).__name__}")
    return errors


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """
    Validate a manifest against the expected schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []
    required_fields = [
        "label",
        "slice",
        "mode",
        "cycles",
        "initial_seed",
        "slice_config_hash",
        "ht_series_hash",
        "outputs",
    ]
    for field in required_fields:
        if field not in manifest:
            errors.append(f"Manifest missing required field: {field}")
    if "outputs" in manifest:
        if "results" not in manifest["outputs"]:
            errors.append("Manifest outputs missing 'results' path")
        if "manifest" not in manifest["outputs"]:
            errors.append("Manifest outputs missing 'manifest' path")
    return errors


def verify_determinism(
    log_path: Path, seed: int, slice_name: str, mode: str, config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Verify determinism by replaying the experiment and comparing results.

    Returns a dict with determinism check results.
    """
    from experiments.run_uplift_u2 import (
        METRIC_DISPATCHER,
        RFLPolicy,
        generate_seed_schedule,
    )

    result = {
        "deterministic": False,
        "original_hash": None,
        "replay_hash": None,
        "errors": [],
    }

    if not log_path.exists():
        result["errors"].append(f"Log file not found: {log_path}")
        return result

    # Read original records
    original_records = []
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    original_records.append(json.loads(line))
    except Exception as e:
        result["errors"].append(f"Failed to read log: {e}")
        return result

    if not original_records:
        result["errors"].append("No records in log file")
        return result

    cycles = len(original_records)
    result["original_hash"] = hash_string(json.dumps(original_records, sort_keys=True))

    # Replay the experiment
    slice_config = config.get("slices", {}).get(slice_name)
    if not slice_config:
        result["errors"].append(f"Slice '{slice_name}' not found in config")
        return result

    items = slice_config["items"]
    success_metric = METRIC_DISPATCHER.get(slice_name)
    if not success_metric:
        result["errors"].append(f"Success metric for slice '{slice_name}' not found")
        return result

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None

    replay_records = []
    for i in range(cycles):
        cycle_seed = seed_schedule[i]
        rng = random.Random(cycle_seed)

        if mode == "baseline":
            ordered_items = list(items)
            rng.shuffle(ordered_items)
            chosen_item = ordered_items[0]
        elif mode == "rfl":
            item_scores = policy.score(items)
            scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
            chosen_item = scored_items[0][0]
        else:
            result["errors"].append(f"Unknown mode: {mode}")
            return result

        # NOTE: eval() mirrors run_uplift_u2.py's mock execution for determinism replay.
        # Input is from controlled config (curriculum_uplift_phase2.yaml), not user input.
        # Only arithmetic_simple slice uses eval; items are simple arithmetic like "1 + 1".
        mock_result = (
            eval(chosen_item)  # nosec B307 - controlled input from config
            if slice_name == "arithmetic_simple"
            else f"Expanded({chosen_item})"
        )
        success = success_metric(chosen_item, mock_result)

        if mode == "rfl":
            policy.update(chosen_item, success)

        record = {
            "cycle": i,
            "slice": slice_name,
            "mode": mode,
            "seed": cycle_seed,
            "item": chosen_item,
            "result": str(mock_result),
            "success": success,
            "label": "PHASE II — NOT USED IN PHASE I",
        }
        replay_records.append(record)

    result["replay_hash"] = hash_string(json.dumps(replay_records, sort_keys=True))
    result["deterministic"] = result["original_hash"] == result["replay_hash"]

    return result


def count_successes(log_path: Path) -> int:
    """Count the number of successful cycles in a log file."""
    count = 0
    if not log_path.exists():
        return count
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if record.get("success"):
                        count += 1
    except Exception:
        pass
    return count


def run_calibration(
    slice_name: str,
    seed: int,
    config_path: Path,
    out_base: Path,
    cycles: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run calibration mode for a single slice.

    This runs both baseline and RFL modes with a small number of cycles,
    validates schemas and manifests, checks determinism, and returns
    a JSON summary with diagnostics.

    Args:
        slice_name: The experiment slice to run
        seed: Initial random seed for deterministic execution
        config_path: Path to the curriculum config file
        out_base: Base output directory (e.g., results/uplift_u2/calibration)
        cycles: Number of cycles (default: CALIBRATION_CYCLES)

    Returns:
        Dictionary with calibration summary and diagnostics
    """
    from experiments.run_uplift_u2 import get_config, run_experiment

    if cycles is None:
        cycles = CALIBRATION_CYCLES

    # Prepare output directory
    out_dir = out_base / slice_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = get_config(config_path)

    summary = {
        "label": "PHASE II — NOT USED IN PHASE I",
        "calibration_mode": True,
        "slice": slice_name,
        "cycles": cycles,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": {
            "success_count": 0,
            "schema_errors": [],
            "manifest_errors": [],
            "determinism": None,
        },
        "rfl": {
            "success_count": 0,
            "schema_errors": [],
            "manifest_errors": [],
            "determinism": None,
        },
        "overall_status": "pending",
        "errors": [],
    }

    # Run baseline mode
    print(f"[CALIBRATION] Running baseline mode for slice '{slice_name}'...")
    try:
        run_experiment(
            slice_name=slice_name,
            cycles=cycles,
            seed=seed,
            mode="baseline",
            out_dir=out_dir,
            config=config,
        )
    except Exception as e:
        summary["errors"].append(f"Baseline run failed: {e}")

    # Run RFL mode
    print(f"[CALIBRATION] Running RFL mode for slice '{slice_name}'...")
    try:
        run_experiment(
            slice_name=slice_name,
            cycles=cycles,
            seed=seed,
            mode="rfl",
            out_dir=out_dir,
            config=config,
        )
    except Exception as e:
        summary["errors"].append(f"RFL run failed: {e}")

    # Validate baseline outputs
    baseline_log = out_dir / f"uplift_u2_{slice_name}_baseline.jsonl"
    baseline_manifest = out_dir / f"uplift_u2_manifest_{slice_name}_baseline.json"

    if baseline_log.exists():
        summary["baseline"]["success_count"] = count_successes(baseline_log)
        # Validate schema of each record
        try:
            with open(baseline_log, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        errs = validate_schema(record)
                        if errs:
                            summary["baseline"]["schema_errors"].extend(
                                [f"Record {i}: {e}" for e in errs]
                            )
        except Exception as e:
            summary["baseline"]["schema_errors"].append(f"Failed to read log: {e}")
    else:
        summary["baseline"]["schema_errors"].append("Baseline log file not found")

    if baseline_manifest.exists():
        try:
            with open(baseline_manifest, "r") as f:
                manifest = json.load(f)
            errs = validate_manifest(manifest)
            summary["baseline"]["manifest_errors"] = errs
        except Exception as e:
            summary["baseline"]["manifest_errors"].append(f"Failed to read manifest: {e}")
    else:
        summary["baseline"]["manifest_errors"].append("Baseline manifest not found")

    # Verify baseline determinism
    summary["baseline"]["determinism"] = verify_determinism(
        baseline_log, seed, slice_name, "baseline", config
    )

    # Validate RFL outputs
    rfl_log = out_dir / f"uplift_u2_{slice_name}_rfl.jsonl"
    rfl_manifest = out_dir / f"uplift_u2_manifest_{slice_name}_rfl.json"

    if rfl_log.exists():
        summary["rfl"]["success_count"] = count_successes(rfl_log)
        try:
            with open(rfl_log, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if line:
                        record = json.loads(line)
                        errs = validate_schema(record)
                        if errs:
                            summary["rfl"]["schema_errors"].extend(
                                [f"Record {i}: {e}" for e in errs]
                            )
        except Exception as e:
            summary["rfl"]["schema_errors"].append(f"Failed to read log: {e}")
    else:
        summary["rfl"]["schema_errors"].append("RFL log file not found")

    if rfl_manifest.exists():
        try:
            with open(rfl_manifest, "r") as f:
                manifest = json.load(f)
            errs = validate_manifest(manifest)
            summary["rfl"]["manifest_errors"] = errs
        except Exception as e:
            summary["rfl"]["manifest_errors"].append(f"Failed to read manifest: {e}")
    else:
        summary["rfl"]["manifest_errors"].append("RFL manifest not found")

    # Verify RFL determinism
    summary["rfl"]["determinism"] = verify_determinism(
        rfl_log, seed, slice_name, "rfl", config
    )

    # Determine overall status
    has_errors = bool(summary["errors"])
    has_baseline_issues = bool(
        summary["baseline"]["schema_errors"]
        or summary["baseline"]["manifest_errors"]
        or (
            summary["baseline"]["determinism"]
            and not summary["baseline"]["determinism"].get("deterministic", False)
        )
    )
    has_rfl_issues = bool(
        summary["rfl"]["schema_errors"]
        or summary["rfl"]["manifest_errors"]
        or (
            summary["rfl"]["determinism"]
            and not summary["rfl"]["determinism"].get("deterministic", False)
        )
    )

    if has_errors or has_baseline_issues or has_rfl_issues:
        summary["overall_status"] = "failed"
    else:
        summary["overall_status"] = "passed"

    # Write summary
    summary_path = out_dir / "calibration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[CALIBRATION] Summary written to {summary_path}")
    print(f"[CALIBRATION] Overall status: {summary['overall_status']}")

    return summary
