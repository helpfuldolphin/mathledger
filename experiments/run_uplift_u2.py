"""
PHASE II -- NOT USED IN PHASE I

U2 Uplift Experiment Runner
===========================

This module runs Phase II uplift experiments. It is designed to be deterministic
and self-contained for reproducibility. Supports two modes:
  - 'baseline': random shuffle ordering (control)
  - 'rfl': policy-driven ordering (treatment)

Module Responsibilities:
  - Load experiment configuration from YAML
  - Execute deterministic experiment cycles
  - Generate telemetry records and manifests
  - Support both baseline and RFL ordering modes

Absolute Safeguards:
  - Do NOT reinterpret Phase I logs as uplift evidence.
  - All Phase II artifacts must be clearly labeled "PHASE II -- NOT USED IN PHASE I".
  - All code must remain deterministic except random shuffle in the baseline policy.
  - RFL uses verifiable feedback only (no RLHF, no preferences, no proxy rewards).
  - All new files must be standalone and MUST NOT modify Phase I behavior.

Exit Codes:
  0: Success
  1: Configuration error (file not found, slice not found)
  2: Validation error (invalid expression, metric not found)
  3: Runtime error (unexpected failure during execution)
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import random
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml

# --- Exit Codes ---
EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 1
EXIT_VALIDATION_ERROR = 2
EXIT_RUNTIME_ERROR = 3

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [U2] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("U2Runner")


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """Structured telemetry record for a single experiment cycle.

    All fields are typed and serializable to JSON. The label field
    explicitly marks this as Phase II data.
    """
    cycle: int
    slice_name: str
    mode: str
    seed: int
    item: str
    result: str
    success: bool
    label: str = field(default="PHASE II -- NOT USED IN PHASE I")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class ExperimentManifest:
    """Manifest capturing experiment configuration and outputs.

    Provides cryptographic hashes for reproducibility verification.
    """
    label: str
    slice_name: str
    mode: str
    cycles: int
    initial_seed: int
    slice_config_hash: str
    prereg_hash: str
    ht_series_hash: str
    deterministic_seed_schedule: List[int]
    results_path: str
    manifest_path: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "label": self.label,
            "slice": self.slice_name,
            "mode": self.mode,
            "cycles": self.cycles,
            "initial_seed": self.initial_seed,
            "slice_config_hash": self.slice_config_hash,
            "prereg_hash": self.prereg_hash,
            "ht_series_hash": self.ht_series_hash,
            "deterministic_seed_schedule": self.deterministic_seed_schedule,
            "outputs": {
                "results": self.results_path,
                "manifest": self.manifest_path,
            }
        }


# -----------------------------------------------------------------------------
# Safe Arithmetic Evaluator
# -----------------------------------------------------------------------------

_SAFE_ARITHMETIC_PATTERN = re.compile(r'^[\d\s\+\-\*\/\(\)\.]+$')


def safe_eval_arithmetic(expr: str) -> Optional[float]:
    """Safely evaluate a simple arithmetic expression.

    Only allows digits, operators (+, -, *, /), parentheses, and whitespace.
    Returns None if the expression is invalid or contains unsafe characters.

    Args:
        expr: Arithmetic expression string (e.g., "2 + 3 * 4")

    Returns:
        Evaluated result as float, or None if invalid/unsafe.
    """
    expr = expr.strip()
    if not expr:
        return None
    if not _SAFE_ARITHMETIC_PATTERN.match(expr):
        logger.debug(f"Rejected unsafe expression: {expr!r}")
        return None
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, (
                ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd
            )):
                logger.debug(f"Rejected expression with disallowed AST node: {type(node).__name__}")
                return None
        result = eval(compile(tree, '<expr>', 'eval'), {"__builtins__": {}}, {})
        return float(result)
    except Exception as e:
        logger.debug(f"Expression evaluation failed: {e}")
        return None


# -----------------------------------------------------------------------------
# Slice-specific Success Metrics
# -----------------------------------------------------------------------------

def metric_arithmetic_simple(item: str, result: Any) -> bool:
    """Success metric for arithmetic_simple slice.

    Uses safe arithmetic evaluation instead of raw eval().
    Success is when the safe evaluation matches the expected result.
    """
    evaluated = safe_eval_arithmetic(item)
    if evaluated is None:
        return False
    try:
        return abs(evaluated - float(result)) < 1e-9
    except (ValueError, TypeError):
        return False


def metric_algebra_expansion(item: str, result: Any) -> bool:
    """Success metric for algebra_expansion slice.

    A mock success metric for algebra. Uses string length comparison
    as a placeholder for real algebraic expansion verification.
    """
    return len(str(result)) > len(item)


METRIC_DISPATCHER: Dict[str, Callable[[str, Any], bool]] = {
    "arithmetic_simple": metric_arithmetic_simple,
    "algebra_expansion": metric_algebra_expansion,
}


# -----------------------------------------------------------------------------
# RFL Policy Implementation
# -----------------------------------------------------------------------------

class RFLPolicy:
    """Mock RFL policy model for Phase II experiments.

    Implements a simple score-based policy with multiplicative updates.
    Scores are initialized lazily and clamped to [0.01, 0.99].

    Attributes:
        scores: Mapping from item to current score.
        rng: Deterministic random number generator.
    """

    def __init__(self, seed: int) -> None:
        """Initialize policy with deterministic seed.

        Args:
            seed: Random seed for reproducible score initialization.
        """
        self.scores: Dict[str, float] = {}
        self.rng = random.Random(seed)

    def score(self, items: List[str]) -> List[float]:
        """Score items for ordering. Higher scores are preferred.

        Args:
            items: List of items to score.

        Returns:
            List of scores in same order as items.
        """
        for item in items:
            if item not in self.scores:
                self.scores[item] = self.rng.random()
        return [self.scores[item] for item in items]

    def update(self, item: str, success: bool) -> None:
        """Update policy based on success feedback.

        Uses multiplicative update rule:
          - Success: score *= 1.1
          - Failure: score *= 0.9

        Args:
            item: The item that was selected.
            success: Whether the item succeeded.
        """
        current = self.scores.get(item, 0.5)
        if success:
            new_score = current * 1.1
        else:
            new_score = current * 0.9
        self.scores[item] = max(0.01, min(new_score, 0.99))


# -----------------------------------------------------------------------------
# Configuration Loading
# -----------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load experiment configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        SystemExit: If config file not found (exit code 1).
    """
    logger.info(f"Loading config from {config_path}")
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(EXIT_CONFIG_ERROR)
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config: {e}")
        sys.exit(EXIT_CONFIG_ERROR)


def validate_slice_config(
    config: Dict[str, Any],
    slice_name: str
) -> Dict[str, Any]:
    """Validate and extract slice configuration.

    Args:
        config: Full configuration dictionary.
        slice_name: Name of the slice to extract.

    Returns:
        Slice configuration dictionary.

    Raises:
        SystemExit: If slice not found (exit code 1).
    """
    slice_config = config.get("slices", {}).get(slice_name)
    if not slice_config:
        logger.error(f"Slice '{slice_name}' not found in config")
        sys.exit(EXIT_CONFIG_ERROR)
    return slice_config


def get_success_metric(slice_name: str) -> Callable[[str, Any], bool]:
    """Get the success metric function for a slice.

    Args:
        slice_name: Name of the slice.

    Returns:
        Success metric function.

    Raises:
        SystemExit: If metric not found (exit code 2).
    """
    metric = METRIC_DISPATCHER.get(slice_name)
    if not metric:
        logger.error(f"Success metric for slice '{slice_name}' not found")
        sys.exit(EXIT_VALIDATION_ERROR)
    return metric


# -----------------------------------------------------------------------------
# Core Experiment Logic
# -----------------------------------------------------------------------------

def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate deterministic seed schedule for experiment cycles.

    Args:
        initial_seed: Initial random seed.
        num_cycles: Number of cycles to generate seeds for.

    Returns:
        List of seeds, one per cycle.
    """
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def hash_string(data: str) -> str:
    """Compute SHA256 hash of a string.

    Args:
        data: String to hash.

    Returns:
        Hex-encoded SHA256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def select_item_baseline(
    items: List[str],
    rng: random.Random
) -> str:
    """Select item using baseline (random shuffle) policy.

    Args:
        items: List of items to select from.
        rng: Random number generator for this cycle.

    Returns:
        Selected item.
    """
    ordered_items = list(items)
    rng.shuffle(ordered_items)
    return ordered_items[0]


def select_item_rfl(
    items: List[str],
    policy: RFLPolicy
) -> str:
    """Select item using RFL policy scoring.

    Args:
        items: List of items to select from.
        policy: RFL policy for scoring.

    Returns:
        Highest-scoring item.
    """
    item_scores = policy.score(items)
    scored_items = sorted(zip(items, item_scores), key=lambda x: x[1], reverse=True)
    return scored_items[0][0]


def execute_item(
    item: str,
    slice_name: str
) -> str:
    """Execute/evaluate an item and return the result.

    For arithmetic_simple: uses safe arithmetic evaluation.
    For other slices: returns a mock expanded result.

    Args:
        item: Item to execute.
        slice_name: Name of the slice (determines evaluation method).

    Returns:
        String representation of the result.
    """
    if slice_name == "arithmetic_simple":
        result = safe_eval_arithmetic(item)
        return str(result) if result is not None else "ERROR"
    else:
        return f"Expanded({item})"


def run_single_cycle(
    cycle_index: int,
    items: List[str],
    mode: str,
    cycle_seed: int,
    policy: Optional[RFLPolicy],
    slice_name: str,
    success_metric: Callable[[str, Any], bool]
) -> TelemetryRecord:
    """Run a single experiment cycle.

    Args:
        cycle_index: Index of this cycle (0-based).
        items: List of items available for selection.
        mode: Execution mode ('baseline' or 'rfl').
        cycle_seed: Deterministic seed for this cycle.
        policy: RFL policy (None for baseline mode).
        slice_name: Name of the experiment slice.
        success_metric: Function to evaluate success.

    Returns:
        TelemetryRecord for this cycle.
    """
    rng = random.Random(cycle_seed)

    # Select item based on mode
    if mode == "baseline":
        chosen_item = select_item_baseline(items, rng)
    elif mode == "rfl":
        chosen_item = select_item_rfl(items, policy)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Execute and evaluate
    result = execute_item(chosen_item, slice_name)
    success = success_metric(chosen_item, result)

    # Update policy if in RFL mode
    if mode == "rfl" and policy is not None:
        policy.update(chosen_item, success)

    return TelemetryRecord(
        cycle=cycle_index,
        slice_name=slice_name,
        mode=mode,
        seed=cycle_seed,
        item=chosen_item,
        result=result,
        success=success
    )


def write_results(
    results_path: Path,
    ht_series: List[TelemetryRecord]
) -> None:
    """Write telemetry results to JSONL file.

    Args:
        results_path: Path to output file.
        ht_series: List of telemetry records.
    """
    with open(results_path, "w", encoding="utf-8") as f:
        for record in ht_series:
            f.write(json.dumps(record.to_dict()) + "\n")


def create_manifest(
    slice_name: str,
    mode: str,
    cycles: int,
    seed: int,
    slice_config: Dict[str, Any],
    ht_series: List[TelemetryRecord],
    seed_schedule: List[int],
    results_path: Path,
    manifest_path: Path
) -> ExperimentManifest:
    """Create experiment manifest with hashes.

    Args:
        slice_name: Name of the experiment slice.
        mode: Execution mode.
        cycles: Number of cycles.
        seed: Initial seed.
        slice_config: Slice configuration dictionary.
        ht_series: Telemetry records.
        seed_schedule: List of cycle seeds.
        results_path: Path to results file.
        manifest_path: Path to manifest file.

    Returns:
        ExperimentManifest instance.
    """
    slice_config_str = json.dumps(slice_config, sort_keys=True)
    ht_series_dicts = [r.to_dict() for r in ht_series]
    ht_series_str = json.dumps(ht_series_dicts, sort_keys=True)

    return ExperimentManifest(
        label="PHASE II -- NOT USED IN PHASE I",
        slice_name=slice_name,
        mode=mode,
        cycles=cycles,
        initial_seed=seed,
        slice_config_hash=hash_string(slice_config_str),
        prereg_hash=slice_config.get("prereg_hash", "N/A"),
        ht_series_hash=hash_string(ht_series_str),
        deterministic_seed_schedule=seed_schedule,
        results_path=str(results_path),
        manifest_path=str(manifest_path)
    )


def write_manifest(manifest_path: Path, manifest: ExperimentManifest) -> None:
    """Write manifest to JSON file.

    Args:
        manifest_path: Path to output file.
        manifest: Manifest to write.
    """
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)


# -----------------------------------------------------------------------------
# Main Experiment Runner
# -----------------------------------------------------------------------------

def run_experiment(
    slice_name: str,
    cycles: int,
    seed: int,
    mode: str,
    out_dir: Path,
    config: Dict[str, Any],
) -> int:
    """Run a complete uplift experiment.

    Args:
        slice_name: Name of the experiment slice.
        cycles: Number of cycles to run.
        seed: Initial random seed.
        mode: Execution mode ('baseline' or 'rfl').
        out_dir: Output directory for results.
        config: Full configuration dictionary.

    Returns:
        Exit code (0 for success).
    """
    logger.info(f"--- Running Experiment: slice={slice_name}, mode={mode}, cycles={cycles}, seed={seed} ---")
    logger.info("PHASE II -- NOT USED IN PHASE I")

    # Setup
    out_dir.mkdir(exist_ok=True, parents=True)
    slice_config = validate_slice_config(config, slice_name)
    items = slice_config["items"]
    success_metric = get_success_metric(slice_name)

    seed_schedule = generate_seed_schedule(seed, cycles)
    policy = RFLPolicy(seed) if mode == "rfl" else None
    ht_series: List[TelemetryRecord] = []

    results_path = out_dir / f"uplift_u2_{slice_name}_{mode}.jsonl"
    manifest_path = out_dir / f"uplift_u2_manifest_{slice_name}_{mode}.json"

    # Execute cycles
    for i in range(cycles):
        try:
            record = run_single_cycle(
                cycle_index=i,
                items=items,
                mode=mode,
                cycle_seed=seed_schedule[i],
                policy=policy,
                slice_name=slice_name,
                success_metric=success_metric
            )
            ht_series.append(record)
            logger.info(f"Cycle {i+1}/{cycles}: Chose '{record.item}', Success: {record.success}")
        except Exception as e:
            logger.error(f"Cycle {i+1} failed: {e}")
            sys.exit(EXIT_RUNTIME_ERROR)

    # Write outputs
    write_results(results_path, ht_series)

    manifest = create_manifest(
        slice_name=slice_name,
        mode=mode,
        cycles=cycles,
        seed=seed,
        slice_config=slice_config,
        ht_series=ht_series,
        seed_schedule=seed_schedule,
        results_path=results_path,
        manifest_path=manifest_path
    )
    write_manifest(manifest_path, manifest)

    logger.info("--- Experiment Complete ---")
    logger.info(f"Results written to {results_path}")
    logger.info(f"Manifest written to {manifest_path}")

    return EXIT_SUCCESS


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

def main() -> int:
    """CLI entry point for U2 uplift runner.

    Returns:
        Exit code.
    """
    parser = argparse.ArgumentParser(
        description="PHASE II U2 Uplift Runner. Must not be used for Phase I.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Absolute Safeguards:
- Do NOT reinterpret Phase I logs as uplift evidence.
- All Phase II artifacts must be clearly labeled.
- RFL uses verifiable feedback only.

Exit Codes:
  0: Success
  1: Configuration error
  2: Validation error
  3: Runtime error
        """
    )
    parser.add_argument(
        "--slice", required=True, type=str,
        help="The experiment slice to run (e.g., 'arithmetic_simple')."
    )
    parser.add_argument(
        "--cycles", required=True, type=int,
        help="Number of experiment cycles to run."
    )
    parser.add_argument(
        "--seed", required=True, type=int,
        help="Initial random seed for deterministic execution."
    )
    parser.add_argument(
        "--mode", required=True, choices=["baseline", "rfl"],
        help="Execution mode: 'baseline' or 'rfl'."
    )
    parser.add_argument(
        "--out", required=True, type=str,
        help="Output directory for results and manifest files."
    )
    parser.add_argument(
        "--config", default="config/curriculum_uplift_phase2.yaml", type=str,
        help="Path to the curriculum config file."
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug logging."
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config)
    out_dir = Path(args.out)

    config = load_config(config_path)

    return run_experiment(
        slice_name=args.slice,
        cycles=args.cycles,
        seed=args.seed,
        mode=args.mode,
        out_dir=out_dir,
        config=config,
    )


if __name__ == "__main__":
    sys.exit(main())
