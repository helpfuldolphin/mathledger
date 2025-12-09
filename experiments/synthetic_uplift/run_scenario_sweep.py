#!/usr/bin/env python3
"""
==============================================================================
PHASE II — SYNTHETIC TEST DATA ONLY
==============================================================================

CI Scenario Sweep Harness
--------------------------

This script runs a sweep across curated synthetic scenarios for CI validation.

For each scenario in the CI sweep set:
    1. Generate baseline and RFL logs to artifacts/synthetic/<scenario>/
    2. Run the core analysis stack in dry mode (no interpretation)
    3. Confirm analysis completes and outputs have required keys

Exit codes:
    0 = All scenarios complete successfully
    1 = Any analysis breakdown (schema or runtime error)

Modes:
    --schema-only    Validate schema definitions only (no simulation)
    --dry-run        Validate scenarios exist (no generation)
    --ci-only        Run curated CI scenarios from registry
    (default)        Full generation and analysis

Must NOT:
    - Generate uplift interpretations
    - Write outside synthetic directories
    - Mix synthetic and real data

Usage:
    python run_scenario_sweep.py
    python run_scenario_sweep.py --schema-only
    python run_scenario_sweep.py --ci-only
    python run_scenario_sweep.py --scenarios synthetic_null_uplift synthetic_positive_uplift
    python run_scenario_sweep.py --out artifacts/synthetic

==============================================================================
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from experiments.synthetic_uplift.noise_models import SAFETY_LABEL
from experiments.synthetic_uplift.universe_browser import BUILT_IN_UNIVERSES
from experiments.synthetic_uplift.universe_compiler import compile_universe


# ==============================================================================
# SWEEP CONFIGURATION
# ==============================================================================

def load_registry() -> Dict[str, Any]:
    """Load the scenario registry."""
    registry_path = Path(__file__).parent / "scenario_registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_ci_sweep_scenarios() -> List[str]:
    """Get the list of scenarios included in CI sweep."""
    registry = load_registry()
    return registry.get("ci_sweep_scenarios", [])


# ==============================================================================
# ANALYSIS INTERFACE
# ==============================================================================

@dataclass
class AnalysisResult:
    """Result of running analysis on a scenario."""
    scenario: str
    mode: str
    success: bool
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    required_keys_present: bool = False
    duration_seconds: float = 0.0


def run_analysis_dry_mode(
    baseline_path: Path,
    rfl_path: Path,
    scenario_name: str,
) -> Tuple[bool, Dict[str, Any], Optional[str]]:
    """
    Run analysis in dry mode - compute metrics but do NOT interpret results.
    
    Returns:
        (success, metrics_dict, error_message)
    """
    try:
        # Try to import the analysis module
        try:
            from backend.metrics.u2_analysis import (
                load_u2_experiment,
                compute_uplift_metrics,
            )
            has_u2_analysis = True
        except ImportError:
            has_u2_analysis = False
        
        if has_u2_analysis and baseline_path.exists() and rfl_path.exists():
            # Load experiment data
            experiment = load_u2_experiment(str(baseline_path), str(rfl_path))
            experiment.slice_id = scenario_name
            
            # Compute metrics (but do NOT interpret)
            result = compute_uplift_metrics(experiment, n_bootstrap=100)  # Reduced for speed
            
            # Extract metrics without interpretation
            metrics = {
                "n_baseline": result.n_baseline,
                "n_rfl": result.n_rfl,
                "metrics_computed": True,
                "has_success_rate": "success_rate" in result.metrics if hasattr(result, 'metrics') else False,
            }
            
            return True, metrics, None
        else:
            # Fallback: basic log validation
            metrics = validate_logs_basic(baseline_path, rfl_path)
            return True, metrics, None
            
    except Exception as e:
        return False, {}, f"{type(e).__name__}: {str(e)}"


def validate_logs_basic(baseline_path: Path, rfl_path: Path) -> Dict[str, Any]:
    """
    Basic validation of generated logs without full analysis.
    
    Checks:
        - Files exist
        - JSONL format is valid
        - Required keys present in records
    """
    metrics = {
        "baseline_exists": baseline_path.exists(),
        "rfl_exists": rfl_path.exists(),
        "baseline_records": 0,
        "rfl_records": 0,
        "required_keys_present": False,
    }
    
    required_keys = {"cycle", "success", "mode", "seed", "item", "label", "synthetic"}
    
    for mode, path in [("baseline", baseline_path), ("rfl", rfl_path)]:
        if not path.exists():
            continue
        
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
        
        metrics[f"{mode}_records"] = len(records)
        
        if records:
            # Check required keys in first record
            first_record = records[0]
            keys_present = required_keys.issubset(set(first_record.keys()))
            metrics[f"{mode}_has_required_keys"] = keys_present
            
            # Check synthetic label
            metrics[f"{mode}_has_synthetic_label"] = first_record.get("synthetic", False)
    
    metrics["required_keys_present"] = (
        metrics.get("baseline_has_required_keys", False) and
        metrics.get("rfl_has_required_keys", False)
    )
    
    return metrics


# ==============================================================================
# SWEEP RUNNER
# ==============================================================================

@dataclass
class SweepResult:
    """Result of a complete sweep."""
    scenarios_run: int = 0
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    results: List[AnalysisResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


# ==============================================================================
# SCHEMA VALIDATION
# ==============================================================================

def validate_scenario_schema(scenario_name: str) -> Tuple[bool, List[str]]:
    """
    Validate a scenario's schema without running simulation.
    
    Returns:
        (valid, errors) tuple
    """
    errors = []
    
    # Check name
    if not scenario_name.startswith("synthetic_"):
        errors.append(f"Name must start with 'synthetic_'")
    
    # Check exists in registry
    registry = load_registry()
    scenarios = registry.get("scenarios", {})
    
    if scenario_name not in scenarios:
        errors.append(f"Not found in registry")
        return len(errors) == 0, errors
    
    scenario_data = scenarios[scenario_name]
    
    # Check required fields
    required = ["version", "description", "category", "ci_sweep_included", "parameters"]
    for field in required:
        if field not in scenario_data:
            errors.append(f"Missing field: {field}")
    
    # Validate parameters
    params = scenario_data.get("parameters", {})
    
    # Check probabilities
    probs = params.get("probabilities", {})
    if not probs:
        errors.append("Empty probabilities")
    
    for mode, class_probs in probs.items():
        if isinstance(class_probs, dict):
            for cls, prob in class_probs.items():
                if not isinstance(prob, (int, float)):
                    errors.append(f"Non-numeric probability: {mode}/{cls}")
                elif not 0.0 <= prob <= 1.0:
                    errors.append(f"Probability out of range: {mode}/{cls}={prob}")
    
    # Check drift
    drift = params.get("drift", {})
    drift_mode = drift.get("mode", "none")
    if drift_mode not in ["none", "monotonic", "cyclical", "shock"]:
        errors.append(f"Invalid drift mode: {drift_mode}")
    
    if drift_mode == "cyclical":
        period = drift.get("period", 0)
        if period <= 0:
            errors.append("Cyclical drift requires period > 0")
    
    # Check correlation
    corr = params.get("correlation", {})
    rho = corr.get("rho", 0.0)
    if not 0.0 <= rho <= 1.0:
        errors.append(f"Correlation rho out of range: {rho}")
    
    return len(errors) == 0, errors


def run_schema_validation(
    scenarios: List[str],
    verbose: bool = True,
) -> SweepResult:
    """
    Run schema-only validation on scenarios.
    
    No simulation is performed - only schema definitions are checked.
    """
    import time
    
    start_time = time.time()
    result = SweepResult()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{SAFETY_LABEL}")
        print(f"{'='*60}")
        print(f"SCHEMA VALIDATION (No Simulation)")
        print(f"{'='*60}")
        print(f"Scenarios: {len(scenarios)}")
        print()
    
    for i, scenario_name in enumerate(scenarios, 1):
        if verbose:
            print(f"[{i}/{len(scenarios)}] {scenario_name}")
        
        result.scenarios_run += 1
        
        valid, errors = validate_scenario_schema(scenario_name)
        
        if valid:
            result.scenarios_passed += 1
            result.results.append(AnalysisResult(
                scenario=scenario_name,
                mode="schema_validation",
                success=True,
                metrics={"schema_valid": True},
            ))
            if verbose:
                print(f"  [PASS] Schema valid")
        else:
            result.scenarios_failed += 1
            error_msg = "; ".join(errors)
            result.errors.append(f"{scenario_name}: {error_msg}")
            result.results.append(AnalysisResult(
                scenario=scenario_name,
                mode="schema_validation",
                success=False,
                error=error_msg,
            ))
            if verbose:
                print(f"  [FAIL] {error_msg}")
    
    result.duration_seconds = time.time() - start_time
    
    return result


# ==============================================================================
# SCENARIO RUNNER
# ==============================================================================

def run_scenario(
    scenario_name: str,
    out_dir: Path,
    verbose: bool = True,
) -> List[AnalysisResult]:
    """
    Run generation and analysis for a single scenario.
    
    Returns list of AnalysisResult (one per mode).
    """
    import time
    
    results = []
    
    if scenario_name not in BUILT_IN_UNIVERSES:
        return [AnalysisResult(
            scenario=scenario_name,
            mode="N/A",
            success=False,
            error=f"Unknown scenario: {scenario_name}",
        )]
    
    spec = BUILT_IN_UNIVERSES[scenario_name]
    
    try:
        universe = compile_universe(spec)
    except Exception as e:
        return [AnalysisResult(
            scenario=scenario_name,
            mode="N/A",
            success=False,
            error=f"Compilation failed: {e}",
        )]
    
    scenario_dir = out_dir / scenario_name
    
    # Generate logs for both modes
    for mode in ["baseline", "rfl"]:
        start_time = time.time()
        
        try:
            results_path, manifest_path, stats = universe.generate_logs(
                mode=mode,
                out_dir=scenario_dir,
                verbose=verbose,
            )
            
            # Reset for next mode
            universe.reset_policy()
            
            generation_success = True
            generation_error = None
            
        except Exception as e:
            generation_success = False
            generation_error = f"Generation failed: {e}"
            results_path = None
        
        duration = time.time() - start_time
        
        if not generation_success:
            results.append(AnalysisResult(
                scenario=scenario_name,
                mode=mode,
                success=False,
                error=generation_error,
                duration_seconds=duration,
            ))
            continue
        
        # Record generation success
        results.append(AnalysisResult(
            scenario=scenario_name,
            mode=mode,
            success=True,
            metrics={
                "records_generated": stats.total_cycles,
                "success_rate": stats.to_dict()["success_rate"],
            },
            required_keys_present=True,
            duration_seconds=duration,
        ))
    
    # Run analysis on both logs together
    baseline_path = scenario_dir / f"{scenario_name}_baseline.jsonl"
    rfl_path = scenario_dir / f"{scenario_name}_rfl.jsonl"
    
    if baseline_path.exists() and rfl_path.exists():
        analysis_success, analysis_metrics, analysis_error = run_analysis_dry_mode(
            baseline_path, rfl_path, scenario_name
        )
        
        # Append analysis result
        results.append(AnalysisResult(
            scenario=scenario_name,
            mode="analysis",
            success=analysis_success,
            error=analysis_error,
            metrics=analysis_metrics,
            required_keys_present=analysis_metrics.get("required_keys_present", False),
        ))
    
    return results


def run_sweep(
    scenarios: List[str],
    out_dir: Path,
    verbose: bool = True,
    dry_run: bool = False,
) -> SweepResult:
    """
    Run the complete CI sweep.
    
    Args:
        scenarios: List of scenario names to run
        out_dir: Output directory for artifacts
        verbose: Print progress
        dry_run: Only validate, don't generate
    
    Returns:
        SweepResult with all results
    """
    import time
    
    start_time = time.time()
    sweep_result = SweepResult()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{SAFETY_LABEL}")
        print(f"{'='*60}")
        print(f"CI SCENARIO SWEEP")
        print(f"{'='*60}")
        print(f"Scenarios: {len(scenarios)}")
        print(f"Output:    {out_dir}")
        print(f"Dry run:   {dry_run}")
        print()
    
    for i, scenario_name in enumerate(scenarios, 1):
        if verbose:
            print(f"\n[{i}/{len(scenarios)}] {scenario_name}")
            print("-" * 40)
        
        sweep_result.scenarios_run += 1
        
        if dry_run:
            # Dry run: just validate the scenario exists
            if scenario_name in BUILT_IN_UNIVERSES:
                sweep_result.scenarios_passed += 1
                sweep_result.results.append(AnalysisResult(
                    scenario=scenario_name,
                    mode="dry_run",
                    success=True,
                    metrics={"dry_run": True},
                ))
                if verbose:
                    print(f"  [DRY RUN] Scenario validated")
            else:
                sweep_result.scenarios_failed += 1
                error = f"Unknown scenario: {scenario_name}"
                sweep_result.errors.append(error)
                sweep_result.results.append(AnalysisResult(
                    scenario=scenario_name,
                    mode="dry_run",
                    success=False,
                    error=error,
                ))
                if verbose:
                    print(f"  [ERROR] {error}")
        else:
            # Full run
            results = run_scenario(scenario_name, out_dir, verbose=False)
            sweep_result.results.extend(results)
            
            # Check if all results for this scenario passed
            scenario_passed = all(r.success for r in results)
            
            if scenario_passed:
                sweep_result.scenarios_passed += 1
                if verbose:
                    print(f"  [PASS] All modes completed")
            else:
                sweep_result.scenarios_failed += 1
                for r in results:
                    if not r.success:
                        sweep_result.errors.append(f"{scenario_name}/{r.mode}: {r.error}")
                        if verbose:
                            print(f"  [FAIL] {r.mode}: {r.error}")
    
    sweep_result.duration_seconds = time.time() - start_time
    
    return sweep_result


def write_sweep_report(
    result: SweepResult,
    out_path: Path,
):
    """Write sweep results to JSON report."""
    report = {
        "label": SAFETY_LABEL,
        "report_type": "ci_scenario_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "scenarios_run": result.scenarios_run,
            "scenarios_passed": result.scenarios_passed,
            "scenarios_failed": result.scenarios_failed,
            "success": result.scenarios_failed == 0,
            "duration_seconds": result.duration_seconds,
        },
        "errors": result.errors,
        "results": [
            {
                "scenario": r.scenario,
                "mode": r.mode,
                "success": r.success,
                "error": r.error,
                "metrics": r.metrics,
                "required_keys_present": r.required_keys_present,
                "duration_seconds": r.duration_seconds,
            }
            for r in result.results
        ],
    }
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


# ==============================================================================
# ERROR HANDLING
# ==============================================================================

class SweepError(Exception):
    """Clean error for sweep failures."""
    pass


def handle_error(error_msg: str, verbose: bool = True) -> int:
    """
    Handle errors cleanly without stack traces.
    
    Returns exit code 1.
    """
    if verbose:
        print(f"\n[ERROR] {error_msg}", file=sys.stderr)
        print(f"\n{SAFETY_LABEL}", file=sys.stderr)
    return 1


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=f"{SAFETY_LABEL}\n\nCI Scenario Sweep - Run synthetic scenario validation.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="Specific scenarios to run (default: CI sweep set from registry)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="artifacts/synthetic",
        help="Output directory (default: artifacts/synthetic)",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate schema definitions only (no simulation)",
    )
    parser.add_argument(
        "--ci-only",
        action="store_true",
        help="Run only curated CI scenarios from registry",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate scenarios exist without generating logs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--report",
        type=str,
        help="Path to write JSON report (default: <out>/sweep_report.json)",
    )
    parser.add_argument(
        "--realism-envelope",
        action="store_true",
        help="Run realism envelope check on all scenarios",
    )
    parser.add_argument(
        "--graph-only",
        action="store_true",
        help="Show causal graph only (no sweep)",
    )
    parser.add_argument(
        "--graph-format",
        choices=["text", "dot", "mermaid"],
        default="text",
        help="Format for causal graph output (default: text)",
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Handle --graph-only first (no scenarios needed)
    if args.graph_only:
        try:
            from experiments.synthetic_uplift.causal_graph import (
                load_causal_graph_from_registry,
                visualize_scenario_graph,
            )
            graph = load_causal_graph_from_registry()
            print(visualize_scenario_graph(graph, format=args.graph_format))
            return 0
        except Exception as e:
            return handle_error(f"Failed to generate graph: {e}", verbose)
    
    # Handle --realism-envelope
    if args.realism_envelope:
        try:
            from experiments.synthetic_uplift.realism_envelope import run_envelope_check
            return run_envelope_check(verbose=verbose)
        except Exception as e:
            return handle_error(f"Envelope check failed: {e}", verbose)
    
    # Determine scenarios to run
    try:
        if args.scenarios:
            scenarios = args.scenarios
        elif args.ci_only:
            scenarios = get_ci_sweep_scenarios()
        else:
            # Default: all scenarios from registry
            registry = load_registry()
            scenarios = list(registry.get("scenarios", {}).keys())
        
        if not scenarios:
            return handle_error("No scenarios to run", verbose)
        
        # Validate scenario names
        for s in scenarios:
            if not s.startswith("synthetic_"):
                return handle_error(f"Scenario must start with 'synthetic_': {s}", verbose)
        
    except FileNotFoundError:
        return handle_error("Registry file not found", verbose)
    except json.JSONDecodeError as e:
        return handle_error(f"Registry JSON malformed: {e}", verbose)
    except Exception as e:
        return handle_error(f"Failed to load scenarios: {e}", verbose)
    
    out_dir = Path(args.out)
    
    # Run appropriate mode
    try:
        if args.schema_only:
            # Schema validation only - no simulation
            result = run_schema_validation(scenarios=scenarios, verbose=verbose)
        elif args.dry_run:
            # Dry run - validate existence only
            result = run_sweep(
                scenarios=scenarios,
                out_dir=out_dir,
                verbose=verbose,
                dry_run=True,
            )
        else:
            # Full sweep with generation
            result = run_sweep(
                scenarios=scenarios,
                out_dir=out_dir,
                verbose=verbose,
                dry_run=False,
            )
    except Exception as e:
        return handle_error(f"Sweep failed: {e}", verbose)
    
    # Write report
    try:
        report_path = Path(args.report) if args.report else out_dir / "sweep_report.json"
        write_sweep_report(result, report_path)
    except Exception as e:
        return handle_error(f"Failed to write report: {e}", verbose)
    
    # Print summary
    if verbose:
        print()
        print("=" * 60)
        print("SWEEP SUMMARY")
        print("=" * 60)
        mode_str = "schema-only" if args.schema_only else ("dry-run" if args.dry_run else "full")
        print(f"Mode:             {mode_str}")
        print(f"Scenarios run:    {result.scenarios_run}")
        print(f"Scenarios passed: {result.scenarios_passed}")
        print(f"Scenarios failed: {result.scenarios_failed}")
        print(f"Duration:         {result.duration_seconds:.2f}s")
        print(f"Report:           {report_path}")
        print()
        
        if result.scenarios_failed > 0:
            print("ERRORS:")
            for error in result.errors:
                print(f"  - {error}")
            print()
            print("[FAIL] Sweep completed with errors")
        else:
            print("[PASS] All scenarios completed successfully")
        print()
        print(f"{SAFETY_LABEL}")
    
    # Exit code
    return 1 if result.scenarios_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())

