# PHASE II — NOT USED IN PHASE I
#
# Calibration module for U2 uplift experiments.
# Provides tidy calibration outputs with determinism verification and schema validation.
#
# Absolute Safeguards:
# - Do NOT reinterpret Phase I logs as uplift evidence.
# - All Phase II artifacts must be clearly labeled "PHASE II — NOT USED IN PHASE I".
# - No uplift fields (no Δp, no p-values, no statistics).
# - Determinism replay failures surface cleanly and do not break output formatting.

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def hash_string(data: str) -> str:
    """Computes the SHA256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generates a deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


class CalibrationRunner:
    """
    Calibration runner for U2 uplift experiments.
    
    Outputs calibration results to:
        results/uplift_u2/calibration/<slice>/
            baseline.jsonl
            rfl.jsonl
            calibration_summary.json
    """

    def __init__(
        self,
        slice_name: str,
        config: Dict[str, Any],
        base_output_dir: Path = Path("results/uplift_u2/calibration"),
        verbose_cycles: bool = False,
    ):
        self.slice_name = slice_name
        self.config = config
        self.base_output_dir = base_output_dir
        self.verbose_cycles = verbose_cycles
        
        # Get slice configuration
        self.slice_config = config.get("slices", {}).get(slice_name)
        if not self.slice_config:
            raise ValueError(f"Slice '{slice_name}' not found in config.")
        
        self.items = self.slice_config["items"]
        
        # Output paths
        self.output_dir = self.base_output_dir / slice_name
        self.baseline_path = self.output_dir / "baseline.jsonl"
        self.rfl_path = self.output_dir / "rfl.jsonl"
        self.summary_path = self.output_dir / "calibration_summary.json"

    def _ensure_output_dir(self) -> None:
        """Creates output directory if it doesn't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _mock_success_metric(self, item: str, result: Any) -> bool:
        """Mock success metric for calibration. Returns True if eval matches."""
        try:
            # For arithmetic_simple, eval the expression
            if self.slice_name == "arithmetic_simple":
                return eval(item) == result
            # For other slices, check string length as a proxy
            return len(str(result)) > len(item)
        except Exception:
            return False

    def _run_mode(
        self,
        mode: str,
        cycles: int,
        seed: int,
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Run a single mode (baseline or rfl) and return statistics.
        
        Returns:
            Dict with 'cycles' and 'successes' counts.
        """
        seed_schedule = generate_seed_schedule(seed, cycles)
        
        # Simple policy for RFL mode
        policy_scores: Dict[str, float] = {}
        policy_rng = random.Random(seed)
        
        successes = 0
        determinism_records: List[str] = []
        
        with open(output_path, "w") as f:
            for i in range(cycles):
                cycle_seed = seed_schedule[i]
                rng = random.Random(cycle_seed)
                
                # Ordering step
                if mode == "baseline":
                    ordered_items = list(self.items)
                    rng.shuffle(ordered_items)
                    chosen_item = ordered_items[0]
                    abstained = 0
                    verified = len(self.items)
                elif mode == "rfl":
                    # Initialize scores if needed
                    for item in self.items:
                        if item not in policy_scores:
                            policy_scores[item] = policy_rng.random()
                    
                    scored_items = sorted(
                        self.items,
                        key=lambda x: policy_scores.get(x, 0.5),
                        reverse=True,
                    )
                    chosen_item = scored_items[0]
                    abstained = 0
                    verified = len(self.items)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # Mock execution
                if self.slice_name == "arithmetic_simple":
                    try:
                        mock_result = eval(chosen_item)
                    except Exception:
                        mock_result = None
                else:
                    mock_result = f"Expanded({chosen_item})"
                
                success = self._mock_success_metric(chosen_item, mock_result)
                if success:
                    successes += 1
                
                # RFL policy update
                if mode == "rfl":
                    if success:
                        policy_scores[chosen_item] = min(
                            policy_scores.get(chosen_item, 0.5) * 1.1, 0.99
                        )
                    else:
                        policy_scores[chosen_item] = max(
                            policy_scores.get(chosen_item, 0.5) * 0.9, 0.01
                        )
                
                # Record for determinism verification
                record = {
                    "cycle": i,
                    "slice": self.slice_name,
                    "mode": mode,
                    "seed": cycle_seed,
                    "item": chosen_item,
                    "result": str(mock_result),
                    "success": success,
                    "label": "PHASE II — NOT USED IN PHASE I",
                }
                
                f.write(json.dumps(record) + "\n")
                determinism_records.append(hash_string(json.dumps(record, sort_keys=True)))
                
                # Verbose logging if enabled
                if self.verbose_cycles:
                    item_hash_prefix = hash_string(chosen_item)[:8]
                    print(
                        f"[cycle={i}] mode={mode} success={success} "
                        f"verified={verified} abstained={abstained} item={item_hash_prefix}"
                    )
        
        return {
            "cycles": cycles,
            "successes": successes,
            "determinism_hash": hash_string("".join(determinism_records)),
        }

    def _verify_determinism(
        self,
        mode: str,
        cycles: int,
        seed: int,
        original_hash: str,
    ) -> bool:
        """
        Verify determinism by replaying the run and comparing hashes.
        
        Returns:
            True if replay produces the same hash, False otherwise.
        """
        seed_schedule = generate_seed_schedule(seed, cycles)
        
        policy_scores: Dict[str, float] = {}
        policy_rng = random.Random(seed)
        
        determinism_records: List[str] = []
        
        for i in range(cycles):
            cycle_seed = seed_schedule[i]
            rng = random.Random(cycle_seed)
            
            if mode == "baseline":
                ordered_items = list(self.items)
                rng.shuffle(ordered_items)
                chosen_item = ordered_items[0]
            elif mode == "rfl":
                for item in self.items:
                    if item not in policy_scores:
                        policy_scores[item] = policy_rng.random()
                
                scored_items = sorted(
                    self.items,
                    key=lambda x: policy_scores.get(x, 0.5),
                    reverse=True,
                )
                chosen_item = scored_items[0]
            else:
                return False
            
            if self.slice_name == "arithmetic_simple":
                try:
                    mock_result = eval(chosen_item)
                except Exception:
                    mock_result = None
            else:
                mock_result = f"Expanded({chosen_item})"
            
            success = self._mock_success_metric(chosen_item, mock_result)
            
            if mode == "rfl":
                if success:
                    policy_scores[chosen_item] = min(
                        policy_scores.get(chosen_item, 0.5) * 1.1, 0.99
                    )
                else:
                    policy_scores[chosen_item] = max(
                        policy_scores.get(chosen_item, 0.5) * 0.9, 0.01
                    )
            
            record = {
                "cycle": i,
                "slice": self.slice_name,
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": str(mock_result),
                "success": success,
                "label": "PHASE II — NOT USED IN PHASE I",
            }
            determinism_records.append(hash_string(json.dumps(record, sort_keys=True)))
        
        replay_hash = hash_string("".join(determinism_records))
        return replay_hash == original_hash

    def _validate_schema(self, jsonl_path: Path) -> bool:
        """
        Validate that JSONL output has correct schema.
        
        Returns:
            True if all records have required fields, False otherwise.
        """
        required_fields = {"cycle", "slice", "mode", "seed", "item", "result", "success", "label"}
        
        try:
            with open(jsonl_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not required_fields.issubset(record.keys()):
                        return False
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    def run_calibration(
        self,
        cycles: int = 50,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run calibration for both baseline and RFL modes.
        
        Args:
            cycles: Number of cycles per mode.
            seed: Initial random seed.
        
        Returns:
            Calibration summary dictionary.
        """
        print(f"--- Calibration: slice={self.slice_name}, cycles={cycles}, seed={seed} ---")
        print("PHASE II — NOT USED IN PHASE I")
        
        self._ensure_output_dir()
        
        # Run baseline
        print(f"\nRunning baseline calibration...")
        baseline_stats = self._run_mode("baseline", cycles, seed, self.baseline_path)
        
        # Run RFL
        print(f"\nRunning RFL calibration...")
        rfl_stats = self._run_mode("rfl", cycles, seed, self.rfl_path)
        
        # Verify determinism
        print(f"\nVerifying determinism...")
        baseline_determinism = False
        rfl_determinism = False
        
        try:
            baseline_determinism = self._verify_determinism(
                "baseline", cycles, seed, baseline_stats["determinism_hash"]
            )
        except Exception as e:
            print(f"WARNING: Baseline determinism verification failed: {e}", file=sys.stderr)
        
        try:
            rfl_determinism = self._verify_determinism(
                "rfl", cycles, seed, rfl_stats["determinism_hash"]
            )
        except Exception as e:
            print(f"WARNING: RFL determinism verification failed: {e}", file=sys.stderr)
        
        determinism_verified = baseline_determinism and rfl_determinism
        if not determinism_verified:
            print(
                f"WARNING: Determinism verification failed "
                f"(baseline={baseline_determinism}, rfl={rfl_determinism})",
                file=sys.stderr,
            )
        
        # Validate schema
        print(f"\nValidating output schema...")
        baseline_schema_valid = self._validate_schema(self.baseline_path)
        rfl_schema_valid = self._validate_schema(self.rfl_path)
        schema_valid = baseline_schema_valid and rfl_schema_valid
        
        if not schema_valid:
            print(
                f"WARNING: Schema validation failed "
                f"(baseline={baseline_schema_valid}, rfl={rfl_schema_valid})",
                file=sys.stderr,
            )
        
        # Generate calibration summary (NO uplift fields)
        summary = {
            "slice": self.slice_name,
            "baseline_cycles": baseline_stats["cycles"],
            "baseline_successes": baseline_stats["successes"],
            "rfl_cycles": rfl_stats["cycles"],
            "rfl_successes": rfl_stats["successes"],
            "determinism_verified": determinism_verified,
            "schema_valid": schema_valid,
            "phase": "PHASE II",
        }
        
        # Write summary
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n--- Calibration Complete ---")
        print(f"Baseline: {self.baseline_path}")
        print(f"RFL: {self.rfl_path}")
        print(f"Summary: {self.summary_path}")
        print(f"Determinism verified: {determinism_verified}")
        print(f"Schema valid: {schema_valid}")
        
        return summary


def run_calibration(
    slice_name: str,
    config_path: Path = Path("config/curriculum_uplift_phase2.yaml"),
    output_dir: Path = Path("results/uplift_u2/calibration"),
    cycles: int = 50,
    seed: int = 42,
    verbose_cycles: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to run calibration.
    
    Args:
        slice_name: Name of the slice to calibrate.
        config_path: Path to curriculum config.
        output_dir: Base output directory.
        cycles: Number of cycles per mode.
        seed: Initial random seed.
        verbose_cycles: Whether to print per-cycle logs.
    
    Returns:
        Calibration summary dictionary.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    runner = CalibrationRunner(
        slice_name=slice_name,
        config=config,
        base_output_dir=output_dir,
        verbose_cycles=verbose_cycles,
    )
    
    return runner.run_calibration(cycles=cycles, seed=seed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Calibration Runner. Must not be used for Phase I.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- No uplift fields (no Δp, no p-values, no statistics).
        """,
    )
    parser.add_argument(
        "--slice",
        required=True,
        type=str,
        help="The experiment slice to calibrate (e.g., 'arithmetic_simple').",
    )
    parser.add_argument(
        "--cycles",
        default=50,
        type=int,
        help="Number of calibration cycles per mode (default: 50).",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Initial random seed for deterministic execution (default: 42).",
    )
    parser.add_argument(
        "--config",
        default="config/curriculum_uplift_phase2.yaml",
        type=str,
        help="Path to the curriculum config file.",
    )
    parser.add_argument(
        "--out",
        default="results/uplift_u2/calibration",
        type=str,
        help="Base output directory for calibration results.",
    )
    parser.add_argument(
        "--verbose-cycles",
        action="store_true",
        help="Print per-cycle logs during calibration.",
    )
    
    args = parser.parse_args()
    
    summary = run_calibration(
        slice_name=args.slice,
        config_path=Path(args.config),
        output_dir=Path(args.out),
        cycles=args.cycles,
        seed=args.seed,
        verbose_cycles=args.verbose_cycles,
    )
    
    print(f"\nCalibration Summary: {json.dumps(summary, indent=2)}")
