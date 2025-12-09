#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Enhanced Synthetic Log Generator (v2)
--------------------------------------

This generator extends the original with:
    - Temporal drift modes (none, monotonic, cyclical, shock)
    - Class-correlation noise (intra-class co-failure)
    - Enhanced manifest with full noise parameters
    - Snapshot reproducibility

NOT derived from real derivations; NOT part of Evidence Pack.
Must NOT generate or simulate uplift conclusions.

Usage:
    # Basic generation (no noise)
    python generate_synthetic_logs_v2.py --scenario synthetic_null_uplift --cycles 500 --out ./out

    # With drift
    python generate_synthetic_logs_v2.py --scenario synthetic_drift_monotonic --cycles 500 --out ./out

    # With correlation
    python generate_synthetic_logs_v2.py --scenario synthetic_correlation_high --cycles 500 --out ./out

    # Custom parameters
    python generate_synthetic_logs_v2.py --scenario synthetic_null_uplift \
        --drift-mode cyclical --drift-amplitude 0.15 --drift-period 100 \
        --correlation-rho 0.3 --cycles 500 --out ./out

==============================================================================
"""

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from experiments.synthetic_uplift.noise_models import (
    SAFETY_LABEL,
    CorrelationConfig,
    DriftConfig,
    DriftMode,
    NoiseConfig,
    NoiseEngine,
)


# ==============================================================================
# ENHANCED OUTCOME GENERATOR
# ==============================================================================

@dataclass
class ScenarioConfig:
    """
    Full configuration for a synthetic scenario.
    """
    name: str
    description: str
    probabilities: Dict[str, Dict[str, float]]  # mode -> class -> prob
    items: List[Dict[str, Any]]
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    expected_direction: str = "null"
    prereg_hash: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "probabilities": self.probabilities,
            "items": self.items,
            "noise": self.noise.to_dict(),
            "expected_direction": self.expected_direction,
            "prereg_hash": self.prereg_hash,
        }


class EnhancedOutcomeGenerator:
    """
    Enhanced outcome generator with drift and correlation support.
    """
    
    def __init__(
        self,
        scenario: ScenarioConfig,
        mode: str,
        seed: int,
        total_cycles: int,
    ):
        self.scenario = scenario
        self.mode = mode
        self.seed = seed
        self.total_cycles = total_cycles
        self.probabilities = scenario.probabilities[mode]
        self.items = scenario.items
        self.rng = random.Random(seed)
        
        # Build item -> class mapping
        self.item_class_map = {
            item["id"]: item["class"] for item in self.items
        }
        
        # Initialize noise engine
        self.noise_engine = NoiseEngine(scenario.noise, seed)
    
    def get_item_ids(self) -> List[str]:
        """Return list of all item IDs."""
        return [item["id"] for item in self.items]
    
    def generate_outcome(
        self,
        item_id: str,
        cycle: int,
        cycle_seed: int,
    ) -> Dict[str, Any]:
        """
        Generate a deterministic outcome with drift and correlation applied.
        
        Returns:
            Dict with outcome details including noise metadata
        """
        item_class = self.item_class_map.get(item_id, "class_a")
        base_prob = self.probabilities.get(item_class, 0.5)
        
        # Apply drift modulation
        drifted_prob = self.noise_engine.apply_drift(
            base_prob, cycle, self.total_cycles
        )
        
        # Generate independent outcome
        combined_seed = cycle_seed ^ hash(item_id)
        local_rng = random.Random(combined_seed)
        roll = local_rng.random()
        independent_success = roll < drifted_prob
        
        # Apply correlation
        final_success = self.noise_engine.apply_correlation(
            independent_success,
            item_class,
            cycle,
            cycle_seed,
            item_id,
        )
        
        return {
            "success": final_success,
            "outcome": "VERIFIED" if final_success else "ABSTAIN",
            "mock_result": {
                "synthetic": True,
                "base_probability": base_prob,
                "drifted_probability": drifted_prob,
                "roll": roll,
                "class": item_class,
                "independent_outcome": independent_success,
                "correlated_outcome": final_success,
                "drift_mode": self.scenario.noise.drift.mode.value,
                "correlation_rho": self.scenario.noise.correlation.rho,
            }
        }


# ==============================================================================
# MANIFEST SPECIFICATION
# ==============================================================================

@dataclass
class SyntheticManifest:
    """
    Full manifest for synthetic data generation.
    
    Hash-committed for reproducibility verification.
    """
    label: str = SAFETY_LABEL
    synthetic: bool = True
    version: str = "2.0"
    
    # Scenario metadata
    scenario_name: str = ""
    scenario_description: str = ""
    mode: str = ""
    cycles: int = 0
    initial_seed: int = 0
    
    # Probability matrix
    probability_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Noise parameters
    drift_config: Dict[str, Any] = field(default_factory=dict)
    correlation_config: Dict[str, Any] = field(default_factory=dict)
    
    # Hashes for integrity
    scenario_config_hash: str = ""
    telemetry_hash: str = ""
    noise_config_hash: str = ""
    
    # Statistics
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    # Outputs
    outputs: Dict[str, str] = field(default_factory=dict)
    
    # Timestamps
    generated_at: str = ""
    seed_schedule: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": self.label,
            "synthetic": self.synthetic,
            "version": self.version,
            "scenario_name": self.scenario_name,
            "scenario_description": self.scenario_description,
            "mode": self.mode,
            "cycles": self.cycles,
            "initial_seed": self.initial_seed,
            "probability_matrix": self.probability_matrix,
            "drift_config": self.drift_config,
            "correlation_config": self.correlation_config,
            "scenario_config_hash": self.scenario_config_hash,
            "telemetry_hash": self.telemetry_hash,
            "noise_config_hash": self.noise_config_hash,
            "statistics": self.statistics,
            "outputs": self.outputs,
            "generated_at": self.generated_at,
            "seed_schedule": self.seed_schedule,
        }
    
    def save(self, path: Path):
        """Save manifest to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


# ==============================================================================
# CORE GENERATION LOGIC
# ==============================================================================

def compute_sha256(data: str) -> str:
    """Compute SHA256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate deterministic list of seeds for each cycle."""
    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def select_item_for_cycle(
    items: List[str],
    mode: str,
    cycle_seed: int,
    policy_scores: Optional[Dict[str, float]] = None,
) -> str:
    """Select an item for this cycle based on mode."""
    rng = random.Random(cycle_seed)
    
    if mode == "baseline":
        return rng.choice(items)
    else:
        if policy_scores:
            scored = sorted(
                [(item, policy_scores.get(item, 0.5)) for item in items],
                key=lambda x: x[1],
                reverse=True,
            )
            return scored[0][0]
        else:
            return rng.choice(items)


def update_policy_scores(
    scores: Dict[str, float],
    item: str,
    success: bool,
) -> Dict[str, float]:
    """Update mock RFL policy scores."""
    new_scores = scores.copy()
    if item not in new_scores:
        new_scores[item] = 0.5
    
    if success:
        new_scores[item] = min(new_scores[item] * 1.1, 0.99)
    else:
        new_scores[item] = max(new_scores[item] * 0.9, 0.01)
    
    return new_scores


def generate_synthetic_logs_v2(
    scenario: ScenarioConfig,
    mode: str,
    cycles: int,
    out_dir: Path,
    seed: int,
    verbose: bool = True,
) -> Tuple[Path, Path]:
    """
    Generate synthetic JSONL logs with full noise model support.
    
    Returns:
        Tuple of (results_path, manifest_path)
    """
    if verbose:
        print(f"{'='*60}")
        print(f"{SAFETY_LABEL}")
        print(f"{'='*60}")
        print(f"Generating synthetic logs (v2):")
        print(f"  Scenario:     {scenario.name}")
        print(f"  Mode:         {mode}")
        print(f"  Cycles:       {cycles}")
        print(f"  Seed:         {seed}")
        print(f"  Drift:        {scenario.noise.drift.mode.value}")
        print(f"  Correlation:  ρ={scenario.noise.correlation.rho}")
        print(f"  Output:       {out_dir}")
        print()
    
    # Setup
    out_dir.mkdir(parents=True, exist_ok=True)
    generator = EnhancedOutcomeGenerator(scenario, mode, seed, cycles)
    seed_schedule = generate_seed_schedule(seed, cycles)
    
    items = generator.get_item_ids()
    policy_scores: Dict[str, float] = {}
    telemetry_series: List[Dict[str, Any]] = []
    
    # Output paths
    results_path = out_dir / f"synthetic_{scenario.name}_{mode}.jsonl"
    manifest_path = out_dir / f"synthetic_{scenario.name}_{mode}_manifest.json"
    
    # Statistics tracking
    success_count = 0
    per_cycle_probs: List[float] = []
    
    # Main generation loop
    with open(results_path, "w", encoding="utf-8") as f:
        for cycle in range(cycles):
            cycle_seed = seed_schedule[cycle]
            
            # Select item
            chosen_item = select_item_for_cycle(
                items=items,
                mode=mode,
                cycle_seed=cycle_seed,
                policy_scores=policy_scores if mode == "rfl" else None,
            )
            
            # Generate outcome
            outcome = generator.generate_outcome(chosen_item, cycle, cycle_seed)
            success = outcome["success"]
            
            if success:
                success_count += 1
            
            per_cycle_probs.append(outcome["mock_result"]["drifted_probability"])
            
            # Update policy for RFL mode
            if mode == "rfl":
                policy_scores = update_policy_scores(
                    policy_scores, chosen_item, success
                )
            
            # Build telemetry record
            record = {
                "cycle": cycle,
                "scenario": scenario.name,
                "slice": scenario.name,  # Alias for compatibility
                "mode": mode,
                "seed": cycle_seed,
                "item": chosen_item,
                "result": json.dumps(outcome["mock_result"]),
                "success": success,
                "outcome": outcome["outcome"],
                "proof_found": success,
                "abstention": not success,
                "drifted_probability": outcome["mock_result"]["drifted_probability"],
                "label": SAFETY_LABEL,
                "synthetic": True,
                "generator_version": "2.0",
            }
            
            telemetry_series.append(record)
            f.write(json.dumps(record) + "\n")
            
            if verbose and ((cycle + 1) % 100 == 0 or cycle == cycles - 1):
                rate = success_count / (cycle + 1) * 100
                avg_prob = sum(per_cycle_probs) / len(per_cycle_probs)
                print(f"  Cycle {cycle + 1}/{cycles}: success={rate:.1f}%, avg_p={avg_prob:.3f}")
    
    # Build manifest
    manifest = SyntheticManifest(
        scenario_name=scenario.name,
        scenario_description=scenario.description,
        mode=mode,
        cycles=cycles,
        initial_seed=seed,
        probability_matrix=scenario.probabilities,
        drift_config=scenario.noise.drift.to_dict(),
        correlation_config=scenario.noise.correlation.to_dict(),
        scenario_config_hash=compute_sha256(json.dumps(scenario.to_dict(), sort_keys=True)),
        telemetry_hash=compute_sha256(json.dumps(telemetry_series, sort_keys=True)),
        noise_config_hash=compute_sha256(json.dumps(scenario.noise.to_dict(), sort_keys=True)),
        statistics={
            "total_cycles": cycles,
            "success_count": success_count,
            "success_rate": success_count / cycles if cycles > 0 else 0.0,
            "abstention_rate": 1.0 - (success_count / cycles) if cycles > 0 else 1.0,
            "mean_drifted_probability": sum(per_cycle_probs) / len(per_cycle_probs) if per_cycle_probs else 0.0,
            "min_drifted_probability": min(per_cycle_probs) if per_cycle_probs else 0.0,
            "max_drifted_probability": max(per_cycle_probs) if per_cycle_probs else 0.0,
        },
        outputs={
            "results": str(results_path),
            "manifest": str(manifest_path),
        },
        generated_at=datetime.now(timezone.utc).isoformat(),
        seed_schedule=seed_schedule,
    )
    
    manifest.save(manifest_path)
    
    if verbose:
        print()
        print(f"{'='*60}")
        print("Generation Complete")
        print(f"{'='*60}")
        print(f"  Results:      {results_path}")
        print(f"  Manifest:     {manifest_path}")
        print(f"  Success rate: {manifest.statistics['success_rate']*100:.2f}%")
        print(f"  Prob range:   [{manifest.statistics['min_drifted_probability']:.3f}, {manifest.statistics['max_drifted_probability']:.3f}]")
        print()
        print(f"⚠️  {SAFETY_LABEL}")
        print()
    
    return results_path, manifest_path


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

def main():
    """CLI entry point for v2 generator."""
    parser = argparse.ArgumentParser(
        description=f"{SAFETY_LABEL}\n\nEnhanced synthetic log generator with drift and correlation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "--scenario",
        required=True,
        type=str,
        help="Scenario name from scenario suite",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "rfl"],
        help="Execution mode",
    )
    parser.add_argument(
        "--cycles",
        required=True,
        type=int,
        help="Number of cycles",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=str,
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    # Drift parameters
    parser.add_argument(
        "--drift-mode",
        choices=["none", "monotonic", "cyclical", "shock"],
        default=None,
        help="Override drift mode",
    )
    parser.add_argument("--drift-amplitude", type=float, default=None)
    parser.add_argument("--drift-period", type=int, default=None)
    parser.add_argument("--drift-slope", type=float, default=None)
    parser.add_argument("--drift-shock-cycle", type=int, default=None)
    parser.add_argument("--drift-shock-delta", type=float, default=None)
    
    # Correlation parameters
    parser.add_argument(
        "--correlation-rho",
        type=float,
        default=None,
        help="Override correlation coefficient",
    )
    
    args = parser.parse_args()
    
    # Validate scenario name
    if not args.scenario.startswith("synthetic_"):
        print(f"ERROR: Scenario name must start with 'synthetic_'", file=sys.stderr)
        sys.exit(1)
    
    # Load scenario from suite
    from experiments.synthetic_uplift.scenario_suite import load_scenario
    
    try:
        scenario = load_scenario(args.scenario)
    except KeyError:
        from experiments.synthetic_uplift.scenario_suite import list_scenarios
        print(f"ERROR: Unknown scenario '{args.scenario}'", file=sys.stderr)
        print(f"Available scenarios: {list_scenarios()}", file=sys.stderr)
        sys.exit(1)
    
    # Apply CLI overrides
    if args.drift_mode:
        scenario.noise.drift.mode = DriftMode(args.drift_mode)
    if args.drift_amplitude is not None:
        scenario.noise.drift.amplitude = args.drift_amplitude
    if args.drift_period is not None:
        scenario.noise.drift.period = args.drift_period
    if args.drift_slope is not None:
        scenario.noise.drift.slope = args.drift_slope
    if args.drift_shock_cycle is not None:
        scenario.noise.drift.shock_cycle = args.drift_shock_cycle
    if args.drift_shock_delta is not None:
        scenario.noise.drift.shock_delta = args.drift_shock_delta
    if args.correlation_rho is not None:
        scenario.noise.correlation.rho = args.correlation_rho
    
    # Generate
    generate_synthetic_logs_v2(
        scenario=scenario,
        mode=args.mode,
        cycles=args.cycles,
        out_dir=Path(args.out),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

