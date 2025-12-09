#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Synthetic Scenario Suite
-------------------------

This module provides 12 ready-made synthetic universes for stress-testing
the U2 uplift analysis pipeline:

    1.  synthetic_null_uplift         - No difference between baseline/RFL
    2.  synthetic_positive_uplift     - RFL outperforms baseline
    3.  synthetic_negative_uplift     - Baseline outperforms RFL (regression)
    4.  synthetic_drift_monotonic_up  - Success rate increases over time
    5.  synthetic_drift_monotonic_down- Success rate decreases over time
    6.  synthetic_drift_cyclical      - Sinusoidal probability oscillation
    7.  synthetic_drift_shock         - Sudden probability collapse
    8.  synthetic_correlation_low     - Low intra-class correlation (ρ=0.2)
    9.  synthetic_correlation_high    - High intra-class correlation (ρ=0.7)
    10. synthetic_adversarial         - RFL policy is actively misled
    11. synthetic_catastrophic        - System collapse after threshold
    12. synthetic_mixed_noise         - Combined drift + correlation

NOT derived from real derivations; NOT part of Evidence Pack.
Must NOT generate or simulate uplift conclusions.

==============================================================================
"""

from typing import Dict, List, Any

from experiments.synthetic_uplift.noise_models import (
    SAFETY_LABEL,
    CorrelationConfig,
    DriftConfig,
    DriftMode,
    NoiseConfig,
    create_no_drift,
    create_monotonic_drift,
    create_cyclical_drift,
    create_shock_drift,
    create_correlation,
)
from experiments.synthetic_uplift.generate_synthetic_logs_v2 import ScenarioConfig


# ==============================================================================
# STANDARD ITEM SET
# ==============================================================================

def make_standard_items(prefix: str, num_per_class: int = 3) -> List[Dict[str, Any]]:
    """
    Generate standard item set with items in each class.
    """
    items = []
    classes = ["class_a", "class_b", "class_c"]
    
    for i, cls in enumerate(classes):
        for j in range(num_per_class):
            items.append({
                "id": f"{prefix}_{cls}_{j:02d}",
                "class": cls,
                "complexity": i + 1,
            })
    
    return items


# ==============================================================================
# SCENARIO DEFINITIONS
# ==============================================================================

SCENARIOS: Dict[str, ScenarioConfig] = {}


# ---------------------------------------------------------------------------
# 1. NULL UPLIFT - No difference
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_null_uplift"] = ScenarioConfig(
    name="synthetic_null_uplift",
    description="Null uplift scenario. Baseline and RFL have identical probabilities.",
    probabilities={
        "baseline": {"class_a": 0.80, "class_b": 0.60, "class_c": 0.40},
        "rfl":      {"class_a": 0.80, "class_b": 0.60, "class_c": 0.40},
    },
    items=make_standard_items("null"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_null_uplift_v2_00000001",
)


# ---------------------------------------------------------------------------
# 2. POSITIVE UPLIFT - RFL wins
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_positive_uplift"] = ScenarioConfig(
    name="synthetic_positive_uplift",
    description="Positive uplift scenario. RFL has +20pp advantage.",
    probabilities={
        "baseline": {"class_a": 0.60, "class_b": 0.40, "class_c": 0.25},
        "rfl":      {"class_a": 0.85, "class_b": 0.65, "class_c": 0.45},
    },
    items=make_standard_items("pos"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="positive",
    prereg_hash="synthetic_positive_uplift_v2_00000002",
)


# ---------------------------------------------------------------------------
# 3. NEGATIVE UPLIFT (REGRESSION) - Baseline wins
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_negative_uplift"] = ScenarioConfig(
    name="synthetic_negative_uplift",
    description="Negative uplift (regression). Baseline has +25pp advantage over RFL.",
    probabilities={
        "baseline": {"class_a": 0.90, "class_b": 0.75, "class_c": 0.60},
        "rfl":      {"class_a": 0.65, "class_b": 0.50, "class_c": 0.35},
    },
    items=make_standard_items("neg"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="negative",
    prereg_hash="synthetic_negative_uplift_v2_00000003",
)


# ---------------------------------------------------------------------------
# 4. DRIFT MONOTONIC UP - Improvement over time
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_drift_monotonic_up"] = ScenarioConfig(
    name="synthetic_drift_monotonic_up",
    description="Monotonic upward drift. Success rate increases by ~20pp over 500 cycles.",
    probabilities={
        "baseline": {"class_a": 0.50, "class_b": 0.40, "class_c": 0.30},
        "rfl":      {"class_a": 0.50, "class_b": 0.40, "class_c": 0.30},
    },
    items=make_standard_items("drift_up"),
    noise=NoiseConfig(
        drift=create_monotonic_drift(slope=0.0004, direction="up"),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_drift_monotonic_up_v2_00000004",
)


# ---------------------------------------------------------------------------
# 5. DRIFT MONOTONIC DOWN - Degradation over time
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_drift_monotonic_down"] = ScenarioConfig(
    name="synthetic_drift_monotonic_down",
    description="Monotonic downward drift. Success rate decreases by ~20pp over 500 cycles.",
    probabilities={
        "baseline": {"class_a": 0.80, "class_b": 0.70, "class_c": 0.60},
        "rfl":      {"class_a": 0.80, "class_b": 0.70, "class_c": 0.60},
    },
    items=make_standard_items("drift_down"),
    noise=NoiseConfig(
        drift=create_monotonic_drift(slope=0.0004, direction="down"),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_drift_monotonic_down_v2_00000005",
)


# ---------------------------------------------------------------------------
# 6. DRIFT CYCLICAL - Oscillating probabilities
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_drift_cyclical"] = ScenarioConfig(
    name="synthetic_drift_cyclical",
    description="Cyclical drift. Sinusoidal oscillation with amplitude ±15%, period 100 cycles.",
    probabilities={
        "baseline": {"class_a": 0.65, "class_b": 0.55, "class_c": 0.45},
        "rfl":      {"class_a": 0.65, "class_b": 0.55, "class_c": 0.45},
    },
    items=make_standard_items("drift_cyc"),
    noise=NoiseConfig(
        drift=create_cyclical_drift(amplitude=0.15, period=100),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_drift_cyclical_v2_00000006",
)


# ---------------------------------------------------------------------------
# 7. DRIFT SHOCK - Sudden collapse
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_drift_shock"] = ScenarioConfig(
    name="synthetic_drift_shock",
    description="Shock drift. Success rate drops by 30pp at cycle 250.",
    probabilities={
        "baseline": {"class_a": 0.75, "class_b": 0.65, "class_c": 0.55},
        "rfl":      {"class_a": 0.75, "class_b": 0.65, "class_c": 0.55},
    },
    items=make_standard_items("drift_shock"),
    noise=NoiseConfig(
        drift=create_shock_drift(shock_cycle=250, shock_delta=-0.30),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_drift_shock_v2_00000007",
)


# ---------------------------------------------------------------------------
# 8. CORRELATION LOW - Mild class correlation
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_correlation_low"] = ScenarioConfig(
    name="synthetic_correlation_low",
    description="Low intra-class correlation (ρ=0.2). Items in same class occasionally co-fail.",
    probabilities={
        "baseline": {"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
        "rfl":      {"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
    },
    items=make_standard_items("corr_low"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=create_correlation(rho=0.2, mode="class"),
    ),
    expected_direction="null",
    prereg_hash="synthetic_correlation_low_v2_00000008",
)


# ---------------------------------------------------------------------------
# 9. CORRELATION HIGH - Strong class correlation
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_correlation_high"] = ScenarioConfig(
    name="synthetic_correlation_high",
    description="High intra-class correlation (ρ=0.7). Items in same class frequently co-fail.",
    probabilities={
        "baseline": {"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
        "rfl":      {"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
    },
    items=make_standard_items("corr_high"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=create_correlation(rho=0.7, mode="class"),
    ),
    expected_direction="null",
    prereg_hash="synthetic_correlation_high_v2_00000009",
)


# ---------------------------------------------------------------------------
# 10. ADVERSARIAL - RFL policy misled
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_adversarial"] = ScenarioConfig(
    name="synthetic_adversarial",
    description="Adversarial scenario. Class probabilities inverted for RFL mode.",
    probabilities={
        "baseline": {"class_a": 0.80, "class_b": 0.50, "class_c": 0.20},
        # RFL sees inverted class success - harder classes succeed, easy fail
        "rfl":      {"class_a": 0.30, "class_b": 0.50, "class_c": 0.70},
    },
    items=make_standard_items("adv"),
    noise=NoiseConfig(
        drift=create_no_drift(),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="negative",  # Baseline should win in aggregate
    prereg_hash="synthetic_adversarial_v2_00000010",
)


# ---------------------------------------------------------------------------
# 11. CATASTROPHIC COLLAPSE
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_catastrophic"] = ScenarioConfig(
    name="synthetic_catastrophic",
    description="Catastrophic collapse. Success drops to near-zero after cycle 200.",
    probabilities={
        "baseline": {"class_a": 0.85, "class_b": 0.75, "class_c": 0.65},
        "rfl":      {"class_a": 0.85, "class_b": 0.75, "class_c": 0.65},
    },
    items=make_standard_items("cata"),
    noise=NoiseConfig(
        drift=create_shock_drift(shock_cycle=200, shock_delta=-0.70),
        correlation=CorrelationConfig(rho=0.0),
    ),
    expected_direction="null",
    prereg_hash="synthetic_catastrophic_v2_00000011",
)


# ---------------------------------------------------------------------------
# 12. MIXED NOISE - Drift + Correlation combined
# ---------------------------------------------------------------------------
SCENARIOS["synthetic_mixed_noise"] = ScenarioConfig(
    name="synthetic_mixed_noise",
    description="Mixed noise. Cyclical drift (±10%, period 150) + moderate correlation (ρ=0.4).",
    probabilities={
        "baseline": {"class_a": 0.65, "class_b": 0.55, "class_c": 0.45},
        "rfl":      {"class_a": 0.70, "class_b": 0.60, "class_c": 0.50},
    },
    items=make_standard_items("mixed"),
    noise=NoiseConfig(
        drift=create_cyclical_drift(amplitude=0.10, period=150),
        correlation=create_correlation(rho=0.4, mode="class"),
    ),
    expected_direction="positive",  # RFL has slight advantage
    prereg_hash="synthetic_mixed_noise_v2_00000012",
)


# ==============================================================================
# SUITE API
# ==============================================================================

def list_scenarios() -> List[str]:
    """Return list of all available scenario names."""
    return list(SCENARIOS.keys())


def load_scenario(name: str) -> ScenarioConfig:
    """
    Load a scenario by name.
    
    Raises:
        KeyError if scenario not found
    """
    if name not in SCENARIOS:
        raise KeyError(f"Unknown scenario: {name}")
    return SCENARIOS[name]


def get_scenario_summary() -> Dict[str, str]:
    """Get summary of all scenarios (name -> description)."""
    return {name: s.description for name, s in SCENARIOS.items()}


def get_scenarios_by_category() -> Dict[str, List[str]]:
    """Group scenarios by category."""
    return {
        "uplift": [
            "synthetic_null_uplift",
            "synthetic_positive_uplift",
            "synthetic_negative_uplift",
        ],
        "drift": [
            "synthetic_drift_monotonic_up",
            "synthetic_drift_monotonic_down",
            "synthetic_drift_cyclical",
            "synthetic_drift_shock",
        ],
        "correlation": [
            "synthetic_correlation_low",
            "synthetic_correlation_high",
        ],
        "adversarial": [
            "synthetic_adversarial",
            "synthetic_catastrophic",
        ],
        "mixed": [
            "synthetic_mixed_noise",
        ],
    }


# ==============================================================================
# BATCH GENERATION
# ==============================================================================

def generate_all_scenarios(
    out_dir_base,
    cycles: int = 500,
    seed: int = 42,
    modes: List[str] = None,
):
    """
    Generate logs for all scenarios in the suite.
    
    Args:
        out_dir_base: Base directory for outputs
        cycles: Number of cycles per scenario
        seed: Random seed
        modes: List of modes to generate (default: ["baseline", "rfl"])
    """
    from pathlib import Path
    from experiments.synthetic_uplift.generate_synthetic_logs_v2 import generate_synthetic_logs_v2
    
    if modes is None:
        modes = ["baseline", "rfl"]
    
    out_base = Path(out_dir_base)
    results = []
    
    for scenario_name in SCENARIOS:
        scenario = SCENARIOS[scenario_name]
        scenario_dir = out_base / scenario_name
        
        for mode in modes:
            print(f"\n{'='*60}")
            print(f"Generating: {scenario_name} / {mode}")
            print(f"{'='*60}")
            
            results_path, manifest_path = generate_synthetic_logs_v2(
                scenario=scenario,
                mode=mode,
                cycles=cycles,
                out_dir=scenario_dir,
                seed=seed,
                verbose=False,
            )
            
            results.append({
                "scenario": scenario_name,
                "mode": mode,
                "results": str(results_path),
                "manifest": str(manifest_path),
            })
            
            print(f"  ✓ Generated {results_path.name}")
    
    # Write index file
    index_path = out_base / "scenario_index.json"
    import json
    with open(index_path, "w") as f:
        json.dump({
            "label": SAFETY_LABEL,
            "generated_scenarios": results,
            "cycles": cycles,
            "seed": seed,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Suite generation complete!")
    print(f"Index: {index_path}")
    print(f"Total: {len(results)} log files")
    print(f"{'='*60}")
    
    return results


# ==============================================================================
# CLI FOR SUITE OPERATIONS
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description=f"{SAFETY_LABEL}\n\nSynthetic scenario suite operations.",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all scenarios")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show scenario details")
    info_parser.add_argument("scenario", help="Scenario name")
    
    # Generate-all command
    gen_parser = subparsers.add_parser("generate-all", help="Generate all scenarios")
    gen_parser.add_argument("--out", required=True, help="Output directory")
    gen_parser.add_argument("--cycles", type=int, default=500, help="Cycles per scenario")
    gen_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "list":
        print(f"\n{SAFETY_LABEL}\n")
        print("Available scenarios:")
        for cat, scenarios in get_scenarios_by_category().items():
            print(f"\n  [{cat.upper()}]")
            for s in scenarios:
                desc = SCENARIOS[s].description[:60] + "..." if len(SCENARIOS[s].description) > 60 else SCENARIOS[s].description
                print(f"    - {s}")
                print(f"      {desc}")
    
    elif args.command == "info":
        if args.scenario not in SCENARIOS:
            print(f"ERROR: Unknown scenario '{args.scenario}'", file=sys.stderr)
            sys.exit(1)
        
        s = SCENARIOS[args.scenario]
        print(f"\n{SAFETY_LABEL}\n")
        print(f"Scenario: {s.name}")
        print(f"Description: {s.description}")
        print(f"Expected Direction: {s.expected_direction}")
        print(f"\nProbabilities:")
        for mode, probs in s.probabilities.items():
            print(f"  {mode}: {probs}")
        print(f"\nDrift: {s.noise.drift.to_dict()}")
        print(f"Correlation: {s.noise.correlation.to_dict()}")
    
    elif args.command == "generate-all":
        from pathlib import Path
        generate_all_scenarios(
            out_dir_base=Path(args.out),
            cycles=args.cycles,
            seed=args.seed,
        )
    
    else:
        parser.print_help()

