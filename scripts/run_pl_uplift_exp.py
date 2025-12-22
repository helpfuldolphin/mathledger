#!/usr/bin/env python3
"""
PL Uplift Experiment - Policy Adaptation Measurement

Measures VERIFIED rate delta under different search-policy configurations
using REAL propositional logic verification (truth-table tautology check).

This is NOT machine learning. It is windowed measurement of a trivial
policy axis (e.g., search breadth) with deterministic verification.

Usage:
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --output results/pl_uplift/
    uv run python scripts/run_pl_uplift_exp.py --seed 42 --cycles 200 --quick

Exit codes:
    0 - Experiment completed (delta positive or negative - both are valid data)
    1 - Configuration/environment error
    2 - Determinism violation detected

Note: This experiment is ADDITIVE to the v0.9.4 audit surface.
It does not modify any governance-critical code paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Imports from existing MathLedger primitives
# ---------------------------------------------------------------------------

try:
    from backend.repro.determinism import (
        SeededRNG,
        deterministic_hash,
        deterministic_timestamp,
    )
    from normalization.taut import truth_table_is_tautology, _extract_atoms
except ImportError as e:
    print(f"ERROR: Missing MathLedger primitive: {e}", file=sys.stderr)
    print("Ensure you are running from the repo root with: uv run python scripts/run_pl_uplift_exp.py", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

EXPERIMENT_VERSION = "0.1.0"
HARNESS_NAME = "pl_uplift_exp"


@dataclass
class ExperimentConfig:
    """Experiment configuration - fully deterministic from seed."""
    seed: int
    output_dir: Path
    cycles: int = 100
    formulas_per_cycle: int = 20
    window_size: int = 10
    # Policy axis: max formula complexity (atom count)
    baseline_max_atoms: int = 2
    adapted_max_atoms: int = 3


@dataclass
class CycleResult:
    """Result of a single experiment cycle."""
    cycle_id: int
    policy_mode: str  # "baseline" or "adapted"
    formulas_attempted: int
    formulas_verified: int
    verified_rate: float
    formula_hashes: List[str]


@dataclass
class ExperimentResult:
    """Complete experiment result."""
    experiment_id: str
    seed: int
    config: Dict[str, Any]
    baseline_cycles: List[Dict[str, Any]]
    adapted_cycles: List[Dict[str, Any]]
    baseline_verified_rate: float
    adapted_verified_rate: float
    delta: float
    delta_positive: bool
    determinism_verified: bool
    timestamp: str


# ---------------------------------------------------------------------------
# Formula Generation (deterministic)
# ---------------------------------------------------------------------------

# Atoms available for formula generation
ATOMS = list("pqrs")

# Connectives with their ASCII representations
CONNECTIVES = {
    "imp": "->",
    "and": "/\\",
    "or": "\\/",
    "not": "~",
}


def generate_formula(rng: SeededRNG, max_atoms: int, max_depth: int = 2) -> str:
    """
    Generate a random propositional formula deterministically.

    Args:
        rng: Seeded RNG for deterministic generation
        max_atoms: Maximum number of distinct atoms to use
        max_depth: Maximum nesting depth

    Returns:
        ASCII formula string
    """
    atoms = ATOMS[:max_atoms]

    def gen_expr(depth: int) -> str:
        if depth >= max_depth or rng.random()[0] < 0.4:
            # Terminal: just an atom
            return rng.choice(atoms)[0]

        # Choose connective
        conn_type = int(rng.randint(0, 4)[0])

        if conn_type == 0:  # Implication
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} -> {right})"
        elif conn_type == 1:  # Conjunction
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} /\\ {right})"
        elif conn_type == 2:  # Disjunction
            left = gen_expr(depth + 1)
            right = gen_expr(depth + 1)
            return f"({left} \\/ {right})"
        else:  # Negation
            inner = gen_expr(depth + 1)
            # Only negate atoms to avoid the double-negation limitation
            if len(inner) == 1:
                return f"~{inner}"
            else:
                return inner

    return gen_expr(0)


def generate_tautology_candidate(rng: SeededRNG, max_atoms: int) -> str:
    """
    Generate a formula more likely to be a tautology.

    Uses known tautology patterns with random atom substitution.
    """
    atoms = ATOMS[:max_atoms]

    # Known tautology templates
    templates = [
        "{a} -> {a}",                           # Identity
        "({a} -> ({b} -> {a}))",                # K axiom
        "(({a} -> {b}) -> (~{b} -> ~{a}))",     # Contraposition
        "(({a} /\\ {b}) -> {a})",               # Conjunction elim left
        "(({a} /\\ {b}) -> {b})",               # Conjunction elim right
        "({a} -> ({a} \\/ {b}))",               # Disjunction intro left
        "({b} -> ({a} \\/ {b}))",               # Disjunction intro right
        "(~{a} -> ({a} -> {b}))",               # Ex falso
        "({a} \\/ ~{a})",                       # Excluded middle
    ]

    template = rng.choice(templates)[0]

    # Substitute atoms
    a = rng.choice(atoms)[0]
    b = rng.choice(atoms)[0]

    return template.format(a=a, b=b)


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class PLUpliftExperiment:
    """PL Uplift experiment runner."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.rng = SeededRNG(config.seed)
        self.experiment_id = f"pl_uplift_{config.seed}"

    def run_cycle(self, cycle_id: int, policy_mode: str, max_atoms: int) -> CycleResult:
        """Run a single cycle with given policy."""
        verified_count = 0
        formula_hashes = []

        for _ in range(self.config.formulas_per_cycle):
            # Mix: 70% random formulas, 30% tautology candidates
            if self.rng.random()[0] < 0.7:
                formula = generate_formula(self.rng, max_atoms)
            else:
                formula = generate_tautology_candidate(self.rng, max_atoms)

            # Real verification via truth table
            is_taut = truth_table_is_tautology(formula)

            if is_taut:
                verified_count += 1
                formula_hashes.append(deterministic_hash(formula)[:16])

        verified_rate = verified_count / self.config.formulas_per_cycle

        return CycleResult(
            cycle_id=cycle_id,
            policy_mode=policy_mode,
            formulas_attempted=self.config.formulas_per_cycle,
            formulas_verified=verified_count,
            verified_rate=verified_rate,
            formula_hashes=formula_hashes,
        )

    def run_phase(self, mode: str, max_atoms: int, num_cycles: int) -> List[CycleResult]:
        """Run a phase (baseline or adapted)."""
        results = []
        for i in range(num_cycles):
            result = self.run_cycle(i, mode, max_atoms)
            results.append(result)
        return results

    def run(self) -> ExperimentResult:
        """Run the complete experiment."""
        print(f"Running PL Uplift Experiment (seed={self.config.seed})")
        print(f"  Cycles per phase: {self.config.cycles}")
        print(f"  Formulas per cycle: {self.config.formulas_per_cycle}")
        print()

        # Phase 1: Baseline (limited atom count)
        print(f"Phase 1: Baseline (max_atoms={self.config.baseline_max_atoms})")
        baseline_results = self.run_phase(
            "baseline",
            self.config.baseline_max_atoms,
            self.config.cycles,
        )
        baseline_rate = sum(r.verified_rate for r in baseline_results) / len(baseline_results)
        print(f"  VERIFIED rate: {baseline_rate:.4f}")

        # Phase 2: Adapted (increased atom count)
        print(f"Phase 2: Adapted (max_atoms={self.config.adapted_max_atoms})")
        adapted_results = self.run_phase(
            "adapted",
            self.config.adapted_max_atoms,
            self.config.cycles,
        )
        adapted_rate = sum(r.verified_rate for r in adapted_results) / len(adapted_results)
        print(f"  VERIFIED rate: {adapted_rate:.4f}")

        # Compute delta
        delta = adapted_rate - baseline_rate
        delta_positive = delta > 0

        print()
        print(f"Delta: {delta:+.4f} ({'positive' if delta_positive else 'negative or zero'})")

        # Verify determinism
        determinism_ok = self._verify_determinism()

        return ExperimentResult(
            experiment_id=self.experiment_id,
            seed=self.config.seed,
            config=asdict(self.config),
            baseline_cycles=[asdict(r) for r in baseline_results],
            adapted_cycles=[asdict(r) for r in adapted_results],
            baseline_verified_rate=baseline_rate,
            adapted_verified_rate=adapted_rate,
            delta=delta,
            delta_positive=delta_positive,
            determinism_verified=determinism_ok,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _verify_determinism(self) -> bool:
        """Verify that re-running with same seed produces same results."""
        # Create fresh RNG with same seed
        rng2 = SeededRNG(self.config.seed)

        # Generate same sequence of formulas
        original_rng = self.rng
        self.rng = rng2

        # Run first cycle of baseline
        result1 = self.run_cycle(0, "baseline", self.config.baseline_max_atoms)

        # Reset and run again
        rng3 = SeededRNG(self.config.seed)
        self.rng = rng3
        result2 = self.run_cycle(0, "baseline", self.config.baseline_max_atoms)

        # Restore
        self.rng = original_rng

        # Check determinism
        match = (
            result1.formulas_verified == result2.formulas_verified and
            result1.formula_hashes == result2.formula_hashes
        )

        if not match:
            print("WARNING: Determinism check failed!")

        return match


# ---------------------------------------------------------------------------
# Output Handling
# ---------------------------------------------------------------------------

def write_results(result: ExperimentResult, output_dir: Path) -> Dict[str, Path]:
    """Write experiment results to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Write full results JSON
    results_path = output_dir / f"pl_uplift_{result.seed}.json"
    result_dict = asdict(result)
    # Convert Path to string for JSON serialization
    result_dict["config"]["output_dir"] = str(result_dict["config"]["output_dir"])
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, default=str)
    paths["results"] = results_path

    # Write summary
    summary_path = output_dir / f"pl_uplift_{result.seed}_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"PL Uplift Experiment Summary\n")
        f.write(f"============================\n\n")
        f.write(f"Experiment ID: {result.experiment_id}\n")
        f.write(f"Seed: {result.seed}\n")
        f.write(f"Timestamp: {result.timestamp}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Cycles: {result.config['cycles']}\n")
        f.write(f"  Formulas/cycle: {result.config['formulas_per_cycle']}\n")
        f.write(f"  Baseline max_atoms: {result.config['baseline_max_atoms']}\n")
        f.write(f"  Adapted max_atoms: {result.config['adapted_max_atoms']}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Baseline VERIFIED rate: {result.baseline_verified_rate:.4f}\n")
        f.write(f"  Adapted VERIFIED rate:  {result.adapted_verified_rate:.4f}\n")
        f.write(f"  Delta: {result.delta:+.4f}\n")
        f.write(f"  Delta positive: {result.delta_positive}\n")
        f.write(f"  Determinism verified: {result.determinism_verified}\n")
    paths["summary"] = summary_path

    # Write manifest for verification
    manifest = {
        "harness": HARNESS_NAME,
        "version": EXPERIMENT_VERSION,
        "seed": result.seed,
        "results_hash": deterministic_hash(json.dumps(result_dict, sort_keys=True)),
        "timestamp": result.timestamp,
    }
    manifest_path = output_dir / f"pl_uplift_{result.seed}_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    paths["manifest"] = manifest_path

    return paths


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="PL Uplift Experiment - Policy Adaptation Measurement"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for deterministic execution"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pl_uplift"),
        help="Output directory (default: results/pl_uplift)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=100,
        help="Number of cycles per phase (default: 100)"
    )
    parser.add_argument(
        "--formulas-per-cycle",
        type=int,
        default=20,
        help="Formulas to test per cycle (default: 20)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 cycles, 10 formulas/cycle"
    )
    args = parser.parse_args()

    # Build config
    if args.quick:
        cycles = 10
        formulas_per_cycle = 10
    else:
        cycles = args.cycles
        formulas_per_cycle = args.formulas_per_cycle

    config = ExperimentConfig(
        seed=args.seed,
        output_dir=args.output,
        cycles=cycles,
        formulas_per_cycle=formulas_per_cycle,
    )

    print("=" * 60)
    print("PL UPLIFT EXPERIMENT - Policy Adaptation Measurement")
    print("=" * 60)
    print()
    print(f"Harness version: {EXPERIMENT_VERSION}")
    print(f"Seed: {config.seed}")
    print(f"Output: {config.output_dir}")
    print()

    # Run experiment
    try:
        experiment = PLUpliftExperiment(config)
        result = experiment.run()
    except Exception as e:
        print(f"\nERROR: Experiment failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Check determinism
    if not result.determinism_verified:
        print("\nFATAL: Determinism violation detected!")
        return 2

    # Write results
    print()
    print("Writing results...")
    paths = write_results(result, config.output_dir)
    for name, path in paths.items():
        print(f"  {name}: {path}")

    # Final verdict
    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"Baseline VERIFIED rate: {result.baseline_verified_rate:.4f}")
    print(f"Adapted VERIFIED rate:  {result.adapted_verified_rate:.4f}")
    print(f"Delta: {result.delta:+.4f}")
    print(f"Determinism: {'VERIFIED' if result.determinism_verified else 'FAILED'}")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Experiment interrupted by user")
        sys.exit(130)
