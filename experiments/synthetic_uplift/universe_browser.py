#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

Universe Browser CLI
---------------------

Command-line interface for browsing, compiling, and generating synthetic
noise universes.

Commands:
    --list                    List all built-in universes
    --show <universe>         Show universe details
    --compile <specfile>      Compile a spec file and validate
    --generate <specfile>     Generate logs from a spec file
    --compare <u1> <u2>       Compare two universes

Must NOT generate uplift interpretations.
All outputs labeled "PHASE II — SYNTHETIC".
Entire system is deterministic.

Usage:
    python universe_browser.py --list
    python universe_browser.py --show synthetic_drift_shock
    python universe_browser.py --compile myspec.yaml
    python universe_browser.py --generate myspec.yaml --out ./output --mode baseline
    python universe_browser.py --compare synthetic_null_uplift synthetic_positive_uplift

==============================================================================
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.noise_schema import (
    NoiseSpec,
    NoiseSpecBuilder,
    RareEventType,
    create_catastrophic_collapse,
    create_class_outlier_burst,
    create_intermittent_failure,
    create_recovery_spike,
    create_sudden_uplift,
    validate_spec,
)
from experiments.synthetic_uplift.universe_compiler import (
    CompilationError,
    compile_universe,
    compare_universes,
)


# ==============================================================================
# BUILT-IN UNIVERSES
# ==============================================================================

def _build_universes() -> Dict[str, NoiseSpec]:
    """Build all built-in universe specifications."""
    universes = {}
    
    # ---------------------------------------------------------------------------
    # 1. NULL UPLIFT
    # ---------------------------------------------------------------------------
    universes["synthetic_null_uplift"] = (
        NoiseSpecBuilder("synthetic_null_uplift", "No difference between baseline and RFL")
        .with_seed(42)
        .with_cycles(500)
        .with_classes(["class_a", "class_b", "class_c"], items_per_class=3)
        .with_probabilities(
            baseline={"class_a": 0.75, "class_b": 0.60, "class_c": 0.45},
            rfl={"class_a": 0.75, "class_b": 0.60, "class_c": 0.45},
        )
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 2. POSITIVE UPLIFT
    # ---------------------------------------------------------------------------
    universes["synthetic_positive_uplift"] = (
        NoiseSpecBuilder("synthetic_positive_uplift", "RFL outperforms baseline by ~20pp")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.60, "class_b": 0.45, "class_c": 0.30},
            rfl={"class_a": 0.80, "class_b": 0.65, "class_c": 0.50},
        )
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 3. NEGATIVE UPLIFT (REGRESSION)
    # ---------------------------------------------------------------------------
    universes["synthetic_negative_uplift"] = (
        NoiseSpecBuilder("synthetic_negative_uplift", "Baseline outperforms RFL (regression)")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.85, "class_b": 0.70, "class_c": 0.55},
            rfl={"class_a": 0.60, "class_b": 0.45, "class_c": 0.30},
        )
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 4. DRIFT MONOTONIC DOWN
    # ---------------------------------------------------------------------------
    universes["synthetic_drift_monotonic"] = (
        NoiseSpecBuilder("synthetic_drift_monotonic", "Linear degradation over time")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.80, "class_b": 0.65, "class_c": 0.50},
        )
        .with_drift(mode="monotonic", slope=0.0005, direction="down")
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 5. DRIFT CYCLICAL
    # ---------------------------------------------------------------------------
    universes["synthetic_drift_cyclical"] = (
        NoiseSpecBuilder("synthetic_drift_cyclical", "Sinusoidal probability oscillation")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.65, "class_b": 0.55, "class_c": 0.45},
        )
        .with_drift(mode="cyclical", amplitude=0.15, period=100)
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 6. DRIFT SHOCK
    # ---------------------------------------------------------------------------
    universes["synthetic_drift_shock"] = (
        NoiseSpecBuilder("synthetic_drift_shock", "Sudden probability drop at cycle 250")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.75, "class_b": 0.65, "class_c": 0.55},
        )
        .with_drift(mode="shock", shock_cycle=250, shock_delta=-0.30)
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 7. CORRELATION LOW
    # ---------------------------------------------------------------------------
    universes["synthetic_correlation_low"] = (
        NoiseSpecBuilder("synthetic_correlation_low", "Low intra-class correlation (rho=0.2)")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
        )
        .with_correlation(rho=0.2, mode="class")
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 8. CORRELATION HIGH
    # ---------------------------------------------------------------------------
    universes["synthetic_correlation_high"] = (
        NoiseSpecBuilder("synthetic_correlation_high", "High intra-class correlation (rho=0.7)")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
        )
        .with_correlation(rho=0.7, mode="class")
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 9. CATASTROPHIC COLLAPSE
    # ---------------------------------------------------------------------------
    universes["synthetic_catastrophic"] = (
        NoiseSpecBuilder("synthetic_catastrophic", "Catastrophic system failure at cycle 200")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.85, "class_b": 0.75, "class_c": 0.65},
        )
        .with_rare_event(create_catastrophic_collapse(
            trigger_cycle=200,
            magnitude=-0.60,
            duration=200,
            recovery_rate=0.0,
        ))
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 10. SUDDEN UPLIFT
    # ---------------------------------------------------------------------------
    universes["synthetic_sudden_uplift"] = (
        NoiseSpecBuilder("synthetic_sudden_uplift", "Breakthrough spike at cycle 150")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.50, "class_b": 0.40, "class_c": 0.30},
        )
        .with_rare_event(create_sudden_uplift(
            trigger_cycle=150,
            magnitude=0.35,
            duration=100,
            recovery_rate=0.02,
        ))
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 11. CLASS OUTLIER BURSTS
    # ---------------------------------------------------------------------------
    universes["synthetic_outlier_bursts"] = (
        NoiseSpecBuilder("synthetic_outlier_bursts", "Sporadic class-specific failures")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.75, "class_b": 0.65, "class_c": 0.55},
        )
        .with_rare_event(create_class_outlier_burst("class_a", trigger_probability=0.03, magnitude=-0.45, duration=5))
        .with_rare_event(create_class_outlier_burst("class_b", trigger_probability=0.02, magnitude=-0.40, duration=4))
        .with_rare_event(create_class_outlier_burst("class_c", trigger_probability=0.01, magnitude=-0.35, duration=3))
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 12. INTERMITTENT FAILURES
    # ---------------------------------------------------------------------------
    universes["synthetic_intermittent"] = (
        NoiseSpecBuilder("synthetic_intermittent", "Random transient failures (5% per cycle)")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.80, "class_b": 0.70, "class_c": 0.60},
        )
        .with_rare_event(create_intermittent_failure(
            trigger_probability=0.05,
            magnitude=-0.50,
            duration=3,
        ))
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 13. VARIANCE NOISE
    # ---------------------------------------------------------------------------
    universes["synthetic_variance"] = (
        NoiseSpecBuilder("synthetic_variance", "Per-cycle and per-item variance")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.70, "class_b": 0.55, "class_c": 0.40},
        )
        .with_variance(per_cycle_sigma=0.05, per_item_sigma=0.03)
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 14. MIXED CHAOS
    # ---------------------------------------------------------------------------
    universes["synthetic_mixed_chaos"] = (
        NoiseSpecBuilder("synthetic_mixed_chaos", "Drift + correlation + rare events combined")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.65, "class_b": 0.55, "class_c": 0.45},
            rfl={"class_a": 0.70, "class_b": 0.60, "class_c": 0.50},
        )
        .with_drift(mode="cyclical", amplitude=0.10, period=150)
        .with_correlation(rho=0.3, mode="class")
        .with_rare_event(create_intermittent_failure(trigger_probability=0.02, magnitude=-0.40, duration=5))
        .with_variance(per_cycle_sigma=0.02)
        .build()
    )
    
    # ---------------------------------------------------------------------------
    # 15. RECOVERY AFTER COLLAPSE
    # ---------------------------------------------------------------------------
    universes["synthetic_recovery"] = (
        NoiseSpecBuilder("synthetic_recovery", "Collapse at 150, recovery spike at 300")
        .with_seed(42)
        .with_cycles(500)
        .with_probabilities(
            baseline={"class_a": 0.75, "class_b": 0.65, "class_c": 0.55},
        )
        .with_rare_event(create_catastrophic_collapse(trigger_cycle=150, magnitude=-0.50, duration=100))
        .with_rare_event(create_recovery_spike(trigger_cycle=300, magnitude=0.40, duration=50, recovery_rate=0.03))
        .build()
    )
    
    return universes


BUILT_IN_UNIVERSES = _build_universes()


# ==============================================================================
# CLI COMMANDS
# ==============================================================================

def cmd_list(args):
    """List all built-in universes."""
    print(f"\n{SAFETY_LABEL}\n")
    print("=" * 60)
    print("BUILT-IN SYNTHETIC UNIVERSES")
    print("=" * 60)
    print()
    
    # Group by category
    categories = {
        "uplift": [],
        "drift": [],
        "correlation": [],
        "rare_events": [],
        "mixed": [],
    }
    
    for name, spec in BUILT_IN_UNIVERSES.items():
        if "uplift" in name and "sudden" not in name:
            categories["uplift"].append((name, spec))
        elif "drift" in name:
            categories["drift"].append((name, spec))
        elif "correlation" in name:
            categories["correlation"].append((name, spec))
        elif any(x in name for x in ["catastrophic", "sudden", "outlier", "intermittent", "recovery"]):
            categories["rare_events"].append((name, spec))
        else:
            categories["mixed"].append((name, spec))
    
    for cat, universes in categories.items():
        if universes:
            print(f"[{cat.upper()}]")
            for name, spec in universes:
                desc = spec.description[:50] + "..." if len(spec.description) > 50 else spec.description
                print(f"  • {name}")
                print(f"    {desc}")
            print()
    
    print(f"Total: {len(BUILT_IN_UNIVERSES)} universes")
    print()


def cmd_show(args):
    """Show details of a universe."""
    name = args.universe
    
    if name not in BUILT_IN_UNIVERSES:
        print(f"ERROR: Unknown universe '{name}'", file=sys.stderr)
        print(f"Use --list to see available universes", file=sys.stderr)
        return 1
    
    spec = BUILT_IN_UNIVERSES[name]
    
    print(f"\n{SAFETY_LABEL}\n")
    print("=" * 60)
    print(f"UNIVERSE: {spec.name}")
    print("=" * 60)
    print()
    print(f"Description: {spec.description}")
    print(f"Version:     {spec.version}")
    print(f"Seed:        {spec.seed}")
    print(f"Cycles:      {spec.num_cycles}")
    print(f"Classes:     {spec.classes}")
    print(f"Items/Class: {spec.num_items_per_class}")
    print(f"Total Items: {len(spec.classes) * spec.num_items_per_class}")
    print()
    
    print("PROBABILITIES:")
    for mode in ["baseline", "rfl"]:
        probs = getattr(spec.probabilities, mode)
        if probs:
            print(f"  {mode}:")
            for cls, prob in probs.items():
                print(f"    {cls}: {prob:.2f}")
    print()
    
    print("DRIFT:")
    drift = spec.drift.to_dict()
    print(f"  Mode:      {drift['mode']}")
    if drift['mode'] != 'none':
        if drift['mode'] == 'monotonic':
            print(f"  Slope:     {drift['slope']}")
            print(f"  Direction: {drift['direction']}")
        elif drift['mode'] == 'cyclical':
            print(f"  Amplitude: {drift['amplitude']}")
            print(f"  Period:    {drift['period']}")
        elif drift['mode'] == 'shock':
            print(f"  Shock at:  cycle {drift['shock_cycle']}")
            print(f"  Delta:     {drift['shock_delta']}")
    print()
    
    print("CORRELATION:")
    corr = spec.correlation.to_dict()
    print(f"  rho:       {corr['rho']}")
    print(f"  Mode:      {corr['mode']}")
    print()
    
    if spec.variance.per_cycle_sigma > 0 or spec.variance.per_item_sigma > 0:
        print("VARIANCE:")
        print(f"  Per-cycle σ: {spec.variance.per_cycle_sigma}")
        print(f"  Per-item σ:  {spec.variance.per_item_sigma}")
        print()
    
    if spec.rare_events:
        print("RARE EVENTS:")
        for i, event in enumerate(spec.rare_events):
            print(f"  [{i+1}] {event.event_type.value}")
            if event.trigger_cycles:
                print(f"      Triggers at: cycles {event.trigger_cycles}")
            if event.trigger_probability > 0:
                print(f"      Probability: {event.trigger_probability*100:.1f}% per cycle")
            print(f"      Magnitude:   {event.magnitude:+.2f}")
            print(f"      Duration:    {event.duration} cycles")
            if event.affected_classes:
                print(f"      Affects:     {event.affected_classes}")
        print()
    
    print(f"Spec Hash: {spec.compute_hash()}")
    print()
    
    return 0


def cmd_compile(args):
    """Compile and validate a spec file."""
    spec_path = Path(args.specfile)
    
    if not spec_path.exists():
        print(f"ERROR: File not found: {spec_path}", file=sys.stderr)
        return 1
    
    print(f"\n{SAFETY_LABEL}\n")
    print(f"Compiling: {spec_path}")
    print()
    
    try:
        spec = NoiseSpec.load(spec_path)
        
        # Validate
        errors = validate_spec(spec)
        if errors:
            print("VALIDATION ERRORS:")
            for err in errors:
                print(f"  ✗ {err}")
            return 1
        
        print("✓ Validation passed")
        
        # Compile
        universe = compile_universe(spec)
        print("✓ Compilation successful")
        print()
        print(f"Universe:   {universe.spec.name}")
        print(f"Spec Hash:  {universe.spec_hash}")
        print(f"Items:      {len(universe.item_ids)}")
        print(f"Cycles:     {universe.spec.num_cycles}")
        print()
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


def cmd_generate(args):
    """Generate logs from a spec file."""
    spec_path = Path(args.specfile)
    out_dir = Path(args.out)
    mode = args.mode
    
    if not spec_path.exists():
        # Check if it's a built-in universe name
        if args.specfile in BUILT_IN_UNIVERSES:
            spec = BUILT_IN_UNIVERSES[args.specfile]
        else:
            print(f"ERROR: File not found: {spec_path}", file=sys.stderr)
            return 1
    else:
        spec = NoiseSpec.load(spec_path)
    
    try:
        universe = compile_universe(spec)
        results_path, manifest_path, stats = universe.generate_logs(
            mode=mode,
            out_dir=out_dir,
            verbose=True,
        )
        return 0
        
    except CompilationError as e:
        print(f"COMPILATION ERROR: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_compare(args):
    """Compare two universes."""
    name1, name2 = args.u1, args.u2
    
    if name1 not in BUILT_IN_UNIVERSES:
        print(f"ERROR: Unknown universe '{name1}'", file=sys.stderr)
        return 1
    if name2 not in BUILT_IN_UNIVERSES:
        print(f"ERROR: Unknown universe '{name2}'", file=sys.stderr)
        return 1
    
    spec1 = BUILT_IN_UNIVERSES[name1]
    spec2 = BUILT_IN_UNIVERSES[name2]
    
    universe1 = compile_universe(spec1)
    universe2 = compile_universe(spec2)
    
    diff = compare_universes(universe1, universe2)
    
    print(f"\n{SAFETY_LABEL}\n")
    print("=" * 60)
    print("UNIVERSE COMPARISON (Structural Only)")
    print("=" * 60)
    print()
    print(f"Universe 1: {name1}")
    print(f"Universe 2: {name2}")
    print()
    
    if not diff["differences"]:
        print("No structural differences found.")
    else:
        print("DIFFERENCES:")
        for key, value in diff["differences"].items():
            print(f"\n  [{key}]")
            if isinstance(value, dict) and "u1" in value and "u2" in value:
                print(f"    U1: {value['u1']}")
                print(f"    U2: {value['u2']}")
            else:
                print(f"    {value}")
    
    print()
    print("NOTE: This is a structural comparison only.")
    print("      No uplift interpretation is provided.")
    print()
    
    return 0


def cmd_export(args):
    """Export a built-in universe to a spec file."""
    name = args.universe
    out_path = Path(args.out)
    
    if name not in BUILT_IN_UNIVERSES:
        print(f"ERROR: Unknown universe '{name}'", file=sys.stderr)
        return 1
    
    spec = BUILT_IN_UNIVERSES[name]
    spec.save(out_path)
    
    print(f"\n{SAFETY_LABEL}\n")
    print(f"Exported: {name}")
    print(f"     To: {out_path}")
    print()
    
    return 0


def cmd_registry(args):
    """Print the scenario registry."""
    registry_path = Path(__file__).parent / "scenario_registry.json"
    
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}", file=sys.stderr)
        return 1
    
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    if args.format == "json":
        print(json.dumps(registry, indent=2))
        return 0
    
    # Text table format
    print(f"\n{SAFETY_LABEL}\n")
    print("=" * 80)
    print("SCENARIO REGISTRY")
    print("=" * 80)
    print(f"Registry Version: {registry.get('registry_version', 'N/A')}")
    print(f"Schema Version:   {registry.get('schema_version', 'N/A')}")
    print(f"Total scenarios:  {len(registry.get('scenarios', {}))}")
    print(f"CI sweep scenarios: {len(registry.get('ci_sweep_scenarios', []))}")
    print()
    
    # Print by category
    scenarios = registry.get("scenarios", {})
    categories = registry.get("categories", {})
    
    for cat_name, cat_info in categories.items():
        cat_scenarios = [
            (name, s) for name, s in scenarios.items()
            if s.get("category") == cat_name
        ]
        
        if cat_scenarios:
            print(f"[{cat_name.upper()}] - {cat_info.get('description', '')}")
            print("-" * 80)
            print(f"{'Name':<35} {'Version':<8} {'CI':<4} Description")
            print("-" * 80)
            
            for name, s in cat_scenarios:
                ci_mark = "Yes" if s.get("ci_sweep_included") else "No"
                desc = s.get("description", "")[:30]
                print(f"{name:<35} {s.get('version', '?'):<8} {ci_mark:<4} {desc}")
            print()
    
    print("=" * 80)
    print("CI SWEEP SCENARIOS:")
    print("-" * 80)
    for name in registry.get("ci_sweep_scenarios", []):
        print(f"  - {name}")
    print()
    
    return 0


def cmd_export_contracts(args):
    """Export machine-readable contracts for scenarios."""
    registry_path = Path(__file__).parent / "scenario_registry.json"
    out_path = Path(args.out)
    
    if not registry_path.exists():
        print(f"ERROR: Registry not found at {registry_path}", file=sys.stderr)
        return 1
    
    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    scenarios = registry.get("scenarios", {})
    
    # Filter by category if specified
    if args.category:
        scenarios = {
            k: v for k, v in scenarios.items()
            if v.get("category") == args.category
        }
    
    # Filter to CI sweep only if specified
    if args.ci_only:
        ci_scenarios = set(registry.get("ci_sweep_scenarios", []))
        scenarios = {k: v for k, v in scenarios.items() if k in ci_scenarios}
    
    # Build contracts
    contracts = {
        "label": SAFETY_LABEL,
        "schema_version": "contract_v2",
        "registry_version": registry.get("registry_version", "unknown"),
        "generated_from": "scenario_registry.json",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": {}
    }
    
    for name, scenario in scenarios.items():
        params = scenario.get("parameters", {})
        probs = params.get("probabilities", {})
        drift = params.get("drift", {})
        corr = params.get("correlation", {})
        rare = params.get("rare_events", [])
        variance = params.get("variance", {})
        
        # Compute probability ranges
        all_probs = []
        for mode_probs in probs.values():
            if isinstance(mode_probs, dict):
                all_probs.extend(mode_probs.values())
        
        contract = {
            "name": name,
            "version": scenario.get("version", "1.0"),
            "category": scenario.get("category"),
            "description": scenario.get("description"),
            "ci_sweep_included": scenario.get("ci_sweep_included", False),
            "probability_ranges": {
                "min": min(all_probs) if all_probs else 0.0,
                "max": max(all_probs) if all_probs else 1.0,
            },
            "drift_characteristics": {
                "mode": drift.get("mode", "none"),
                "has_temporal_drift": drift.get("mode", "none") != "none",
                "amplitude": drift.get("amplitude"),
                "period": drift.get("period"),
                "slope": drift.get("slope"),
                "shock_cycle": drift.get("shock_cycle"),
                "shock_delta": drift.get("shock_delta"),
            },
            "correlation_settings": {
                "rho": corr.get("rho", 0.0),
                "mode": corr.get("mode", "class"),
                "has_correlation": corr.get("rho", 0.0) > 0,
            },
            "variance_settings": {
                "per_cycle_sigma": variance.get("per_cycle_sigma", 0.0),
                "per_item_sigma": variance.get("per_item_sigma", 0.0),
                "has_variance": variance.get("per_cycle_sigma", 0.0) > 0 or variance.get("per_item_sigma", 0.0) > 0,
            },
            "rare_event_definitions": [
                {
                    "type": e.get("type"),
                    "trigger_cycle": e.get("trigger_cycle"),
                    "trigger_probability": e.get("trigger_probability"),
                    "magnitude": e.get("magnitude"),
                    "duration": e.get("duration"),
                    "affected_class": e.get("affected_class"),
                }
                for e in rare
            ],
            "has_rare_events": len(rare) > 0,
        }
        
        contracts["scenarios"][name] = contract
    
    # Save contracts
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(contracts, f, indent=2)
    
    print(f"\n{SAFETY_LABEL}\n")
    print(f"Exported contracts for {len(contracts['scenarios'])} scenarios")
    print(f"Output: {out_path}")
    print()
    
    return 0


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"{SAFETY_LABEL}\n\nUniverse Browser - Browse, compile, and generate synthetic noise universes.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # list
    list_parser = subparsers.add_parser("list", help="List all built-in universes")
    
    # show
    show_parser = subparsers.add_parser("show", help="Show universe details")
    show_parser.add_argument("universe", help="Universe name")
    
    # compile
    compile_parser = subparsers.add_parser("compile", help="Compile and validate a spec file")
    compile_parser.add_argument("specfile", help="Path to spec file (.yaml or .json)")
    
    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate logs from a spec")
    gen_parser.add_argument("specfile", help="Path to spec file or built-in universe name")
    gen_parser.add_argument("--out", required=True, help="Output directory")
    gen_parser.add_argument("--mode", required=True, choices=["baseline", "rfl"], help="Generation mode")
    
    # compare
    cmp_parser = subparsers.add_parser("compare", help="Compare two universes")
    cmp_parser.add_argument("u1", help="First universe name")
    cmp_parser.add_argument("u2", help="Second universe name")
    
    # export
    exp_parser = subparsers.add_parser("export", help="Export a built-in universe to file")
    exp_parser.add_argument("universe", help="Universe name to export")
    exp_parser.add_argument("--out", required=True, help="Output file path (.yaml or .json)")
    
    # registry
    reg_parser = subparsers.add_parser("registry", help="Print the scenario registry")
    reg_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    
    # export-contracts
    contracts_parser = subparsers.add_parser("export-contracts", help="Export machine-readable contracts")
    contracts_parser.add_argument("--out", required=True, help="Output file path (.json)")
    contracts_parser.add_argument("--category", help="Filter by category")
    contracts_parser.add_argument("--ci-only", action="store_true", help="Only CI sweep scenarios")
    
    args = parser.parse_args()
    
    if args.command == "list":
        return cmd_list(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "compile":
        return cmd_compile(args)
    elif args.command == "generate":
        return cmd_generate(args)
    elif args.command == "compare":
        return cmd_compare(args)
    elif args.command == "export":
        return cmd_export(args)
    elif args.command == "registry":
        return cmd_registry(args)
    elif args.command == "export-contracts":
        return cmd_export_contracts(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

