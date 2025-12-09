"""
CycleRunner Performance Profiling Harness
==========================================

PERF ONLY — NO BEHAVIOR CHANGE

Microbenchmark harness for CycleRunner hotspot validation and optimization measurement.
This tool is the foundation of the performance ratchet for CI integration.

Usage:
    # Run baseline benchmark (with optimizations disabled):
    uv run python experiments/profile_cycle_runner.py --tag=baseline --cycles=50

    # Run optimized benchmark (with optimizations enabled, default):
    uv run python experiments/profile_cycle_runner.py --tag=optimized --cycles=50

    # Compare results:
    uv run python experiments/profile_cycle_runner.py --compare

    # Use with CI threshold check:
    uv run python experiments/verify_perf_equivalence.py \\
        --baseline results/perf/baseline.json \\
        --optimized results/perf/optimized.json \\
        --min-improvement 0.1

Outputs:
    results/perf/{tag}.json       - Machine-readable benchmark results
    results/perf/{tag}.prof       - cProfile binary data  
    results/perf/{tag}.txt        - Human-readable summary

Machine-Readable JSON Schema:
    {
        "tag": "baseline|optimized",
        "total_time_s": <float>,
        "cycles": <int>,
        "avg_time_per_cycle_ms": <float>,
        "min_time_per_cycle_ms": <float>,
        "max_time_per_cycle_ms": <float>,
        "timestamp": "<ISO8601>",
        "config": {
            "slice_name": "<str>",
            "mode": "<str>",
            "warmup_cycles": <int>,
            "perf_opt_enabled": <bool>
        },
        "top_functions": [...]
    }

Environment Variables:
    MATHLEDGER_PERF_OPT: Set to "0" to disable performance optimizations
                         (default: "1" = enabled)
"""

import argparse
import cProfile
import os
import pstats
import io
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to sys.path (at position 0 to take precedence)
# This ensures the main 'substrate' package is found, not experiments/substrate.py
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Remove experiments dir from path if present (avoid substrate.py shadowing)
experiments_dir = str(Path(__file__).resolve().parent)
if experiments_dir in sys.path:
    sys.path.remove(experiments_dir)


# Output directory for perf results
PERF_RESULTS_DIR = Path("results") / "perf"


@dataclass
class BenchmarkConfig:
    """Configuration used for a benchmark run."""
    slice_name: str
    mode: str
    warmup_cycles: int
    perf_opt_enabled: bool


@dataclass
class ComponentTiming:
    """Timing data for a logical component."""
    name: str
    avg_ms: float
    total_time_s: float
    calls: int
    functions: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """
    Machine-readable benchmark result for CI integration.
    
    This schema is designed for automated perf ratchet checks.
    Includes component-level breakdown for granular analysis.
    """
    tag: str
    total_time_s: float
    cycles: int
    avg_time_per_cycle_ms: float
    min_time_per_cycle_ms: float
    max_time_per_cycle_ms: float
    timestamp: str
    config: Dict[str, Any]
    top_functions: List[Dict[str, Any]] = field(default_factory=list)
    cycle_times_ms: List[float] = field(default_factory=list)
    components: Dict[str, Dict[str, Any]] = field(default_factory=dict)


def run_benchmark(
    tag: str = "baseline",
    warmup_cycles: int = 10,
    measured_cycles: int = 50,
    slice_name: str = "slice_medium",
    mode: str = "baseline",
    system: str = "pl",
    use_optimized: bool = True,
) -> BenchmarkResult:
    """
    Run CycleRunner benchmark with profiling.
    
    Args:
        tag: Identifier for this benchmark run (e.g., "baseline", "optimized")
        warmup_cycles: Number of warm-up cycles (discarded from timing)
        measured_cycles: Number of cycles to measure
        slice_name: Curriculum slice to use
        mode: Execution mode ("baseline" or "rfl")
        system: System slug
        use_optimized: If True, enable MATHLEDGER_PERF_OPT; if False, disable it
        
    Returns:
        BenchmarkResult with timing and profiling data
    """
    from datetime import datetime
    
    # Set the performance optimization flag BEFORE importing CycleRunner
    # This ensures the flag takes effect when derivation/pipeline.py is loaded
    if use_optimized:
        os.environ["MATHLEDGER_PERF_OPT"] = "1"
    else:
        os.environ["MATHLEDGER_PERF_OPT"] = "0"
    
    # Import after setting env var
    from experiments.run_fo_cycles import CycleRunner
    
    # Ensure output directories exist
    PERF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = PERF_RESULTS_DIR / f"benchmark_{tag}.jsonl"
    
    perf_opt_enabled = os.environ.get("MATHLEDGER_PERF_OPT", "1") == "1"
    
    print(f"=" * 70)
    print(f"CycleRunner Performance Benchmark — {tag.upper()}")
    print(f"=" * 70)
    print(f"Configuration:")
    print(f"  Slice: {slice_name}")
    print(f"  Mode: {mode}")
    print(f"  Warmup cycles: {warmup_cycles}")
    print(f"  Measured cycles: {measured_cycles}")
    print(f"  MATHLEDGER_PERF_OPT: {perf_opt_enabled}")
    print(f"=" * 70)
    
    # Initialize runner
    runner = CycleRunner(mode, output_path, slice_name=slice_name, system=system)
    
    # Phase 1: Warm-up (not timed, not profiled)
    print(f"\nPhase 1: Warm-up ({warmup_cycles} cycles)...")
    for i in range(warmup_cycles):
        runner.run_cycle(i)
        if (i + 1) % 5 == 0:
            sys.stdout.write(f"\r  Warmup: {i + 1}/{warmup_cycles}")
            sys.stdout.flush()
    print(f"\r  Warmup complete: {warmup_cycles} cycles")
    
    # Phase 2: Measured cycles with profiling
    print(f"\nPhase 2: Measured cycles ({measured_cycles} cycles)...")
    
    profiler = cProfile.Profile()
    cycle_times: List[float] = []
    
    profiler.enable()
    overall_start = time.perf_counter()
    
    for i in range(measured_cycles):
        cycle_start = time.perf_counter()
        runner.run_cycle(warmup_cycles + i)  # Continue from warmup index
        cycle_end = time.perf_counter()
        cycle_times.append((cycle_end - cycle_start) * 1000)  # ms
        
        if (i + 1) % 10 == 0:
            avg_so_far = sum(cycle_times) / len(cycle_times)
            sys.stdout.write(f"\r  Measured: {i + 1}/{measured_cycles} (avg: {avg_so_far:.2f}ms)")
            sys.stdout.flush()
    
    overall_end = time.perf_counter()
    profiler.disable()
    
    total_time = overall_end - overall_start
    avg_time = sum(cycle_times) / len(cycle_times)
    min_time = min(cycle_times)
    max_time = max(cycle_times)
    
    print(f"\r  Measured complete: {measured_cycles} cycles")
    print(f"\n" + "=" * 70)
    print(f"Results Summary:")
    print(f"  Total time: {total_time:.3f}s")
    print(f"  Avg cycle time: {avg_time:.2f}ms")
    print(f"  Min cycle time: {min_time:.2f}ms")
    print(f"  Max cycle time: {max_time:.2f}ms")
    print(f"=" * 70)
    
    # Save profiler binary data
    prof_path = PERF_RESULTS_DIR / f"{tag}.prof"
    profiler.dump_stats(str(prof_path))
    print(f"\nProfile data saved to: {prof_path}")
    
    # Generate human-readable report
    string_io = io.StringIO()
    stats = pstats.Stats(profiler, stream=string_io)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(30)  # Top 30 functions
    
    report = string_io.getvalue()
    
    # Parse top functions for structured output
    top_functions = _parse_pstats_output(report, limit=50)  # Get more for component aggregation
    
    # Aggregate into component-level metrics
    components = _aggregate_components(top_functions, measured_cycles)
    
    txt_path = PERF_RESULTS_DIR / f"{tag}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"CycleRunner Performance Profile — {tag.upper()}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Slice: {slice_name}\n")
        f.write(f"  Mode: {mode}\n")
        f.write(f"  Warmup cycles: {warmup_cycles}\n")
        f.write(f"  Measured cycles: {measured_cycles}\n")
        f.write(f"  MATHLEDGER_PERF_OPT: {perf_opt_enabled}\n\n")
        f.write(f"Timing Summary:\n")
        f.write(f"  Total time: {total_time:.3f}s\n")
        f.write(f"  Avg cycle time: {avg_time:.2f}ms\n")
        f.write(f"  Min cycle time: {min_time:.2f}ms\n")
        f.write(f"  Max cycle time: {max_time:.2f}ms\n\n")
        f.write(f"Per-cycle times (ms):\n")
        f.write(f"  {cycle_times}\n\n")
        f.write(f"Profile (Top 30 by cumulative time):\n")
        f.write(f"{'=' * 70}\n")
        f.write(report)
    
    print(f"Report saved to: {txt_path}")
    
    # Build config dict
    config = {
        "slice_name": slice_name,
        "mode": mode,
        "warmup_cycles": warmup_cycles,
        "perf_opt_enabled": perf_opt_enabled,
    }
    
    result = BenchmarkResult(
        tag=tag,
        total_time_s=total_time,
        cycles=measured_cycles,
        avg_time_per_cycle_ms=avg_time,
        min_time_per_cycle_ms=min_time,
        max_time_per_cycle_ms=max_time,
        timestamp=datetime.utcnow().isoformat() + "Z",
        config=config,
        top_functions=top_functions[:20],  # Keep top 20 for human readability
        cycle_times_ms=cycle_times,
        components=components,
    )
    
    # Save machine-readable JSON result
    json_path = PERF_RESULTS_DIR / f"{tag}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2)
    print(f"JSON result saved to: {json_path}")
    
    return result


def _parse_pstats_output(report: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Parse pstats text output into structured function list.
    
    Returns list of dicts with: ncalls, tottime, cumtime, filename, function
    """
    functions = []
    lines = report.split('\n')
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Look for the header line that starts the function table
        if 'ncalls' in line and 'tottime' in line and 'cumtime' in line:
            in_table = True
            continue
        
        if not in_table:
            continue
        
        # Parse function lines (format: ncalls tottime percall cumtime percall filename:lineno(function))
        parts = line.split()
        if len(parts) >= 6:
            try:
                ncalls_str = parts[0]
                # Handle recursive calls like "123/45"
                if '/' in ncalls_str:
                    ncalls = int(ncalls_str.split('/')[0])
                else:
                    ncalls = int(ncalls_str)
                
                tottime = float(parts[1])
                cumtime = float(parts[3])
                location = parts[5] if len(parts) > 5 else ""
                
                # Extract filename and function name
                if ':' in location and '(' in location:
                    file_part = location.split('(')[0]
                    func_part = location.split('(')[1].rstrip(')')
                else:
                    file_part = location
                    func_part = ""
                
                functions.append({
                    "ncalls": ncalls,
                    "tottime": tottime,
                    "cumtime": cumtime,
                    "location": location,
                    "filename": file_part,
                    "function": func_part,
                })
                
                if len(functions) >= limit:
                    break
            except (ValueError, IndexError):
                continue
    
    return functions


# Component classification rules: maps (filename_pattern, func_pattern) -> component_name
COMPONENT_CLASSIFICATION_RULES: List[tuple] = [
    # Scoring-related functions
    (r"pipeline", r"candidate_score", "scoring"),
    (r"pipeline", r"_choose_candidate", "scoring"),
    (r"structure", r"formula_depth", "scoring"),
    
    # Derivation/MP-related functions
    (r"pipeline", r"run_step", "derivation"),
    (r"pipeline", r"_run_mp", "derivation"),
    (r"pipeline", r"_generate_axioms", "derivation"),
    (r"canon", r"normalize", "derivation"),
    (r"structure", r"is_implication", "derivation"),
    (r"structure", r"implication_parts", "derivation"),
    
    # Verification/proof-related functions
    (r"pipeline", r"verify_formula", "verification"),
    (r"truth_table", r"", "verification"),
    (r"lean", r"", "verification"),
    
    # Persistence/IO-related functions
    (r"attestation", r"", "persistence"),
    (r"hash", r"sha256", "persistence"),
    (r"json", r"", "persistence"),
    
    # Policy/RFL-related functions
    (r"pipeline", r"_update_policy", "policy"),
    (r"pipeline", r"policy", "policy"),
]


def _classify_function_to_component(filename: str, funcname: str) -> str:
    """
    Classify a function into a logical component based on filename/function patterns.
    
    Returns component name or "other" if no rule matches.
    """
    import re
    
    filename_lower = filename.lower()
    funcname_lower = funcname.lower()
    
    for file_pattern, func_pattern, component in COMPONENT_CLASSIFICATION_RULES:
        file_match = re.search(file_pattern, filename_lower) if file_pattern else True
        func_match = re.search(func_pattern, funcname_lower) if func_pattern else True
        
        if file_match and func_match:
            return component
    
    return "other"


def _aggregate_components(
    top_functions: List[Dict[str, Any]],
    total_cycles: int,
) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate function timings into component-level metrics.
    
    Args:
        top_functions: List of function timing dicts from pstats
        total_cycles: Number of cycles measured (for avg calculation)
        
    Returns:
        Dict mapping component name -> {avg_ms, total_time_s, calls, functions}
    """
    components: Dict[str, Dict[str, Any]] = {}
    
    for func in top_functions:
        filename = func.get("filename", "")
        funcname = func.get("function", "")
        component = _classify_function_to_component(filename, funcname)
        
        if component not in components:
            components[component] = {
                "avg_ms": 0.0,
                "total_time_s": 0.0,
                "calls": 0,
                "functions": [],
            }
        
        # Use tottime (time spent in function itself, not children)
        # for more accurate component attribution
        tottime = func.get("tottime", 0.0)
        ncalls = func.get("ncalls", 0)
        
        components[component]["total_time_s"] += tottime
        components[component]["calls"] += ncalls
        components[component]["functions"].append(f"{filename}:{funcname}")
    
    # Calculate avg_ms per cycle for each component
    if total_cycles > 0:
        for comp_data in components.values():
            comp_data["avg_ms"] = (comp_data["total_time_s"] / total_cycles) * 1000
            # Round for cleaner output
            comp_data["avg_ms"] = round(comp_data["avg_ms"], 2)
            comp_data["total_time_s"] = round(comp_data["total_time_s"], 4)
    
    return components


def compare_benchmarks(baseline_tag: str = "baseline", optimized_tag: str = "optimized"):
    """
    Compare two benchmark runs and report differences.
    """
    baseline_path = PERF_RESULTS_DIR / f"{baseline_tag}.json"
    optimized_path = PERF_RESULTS_DIR / f"{optimized_tag}.json"
    
    if not baseline_path.exists():
        print(f"ERROR: Baseline results not found: {baseline_path}")
        print("Run with --tag=baseline first")
        return 1
    
    if not optimized_path.exists():
        print(f"ERROR: Optimized results not found: {optimized_path}")
        print("Run with --tag=optimized first")
        return 1
    
    with open(baseline_path, 'r') as f:
        baseline = json.load(f)
    
    with open(optimized_path, 'r') as f:
        optimized = json.load(f)
    
    print("=" * 70)
    print("BENCHMARK COMPARISON: Baseline vs Optimized")
    print("=" * 70)
    print()
    
    # Timing comparison
    baseline_avg = baseline["avg_time_per_cycle_ms"]
    optimized_avg = optimized["avg_time_per_cycle_ms"]
    speedup = (baseline_avg - optimized_avg) / baseline_avg * 100
    
    print("TIMING COMPARISON:")
    print(f"  {'Metric':<25} {'Baseline':>12} {'Optimized':>12} {'Change':>12}")
    print(f"  {'-' * 61}")
    print(f"  {'Avg cycle time (ms)':<25} {baseline_avg:>12.2f} {optimized_avg:>12.2f} {speedup:>+11.1f}%")
    print(f"  {'Min cycle time (ms)':<25} {baseline['min_time_per_cycle_ms']:>12.2f} {optimized['min_time_per_cycle_ms']:>12.2f}")
    print(f"  {'Max cycle time (ms)':<25} {baseline['max_time_per_cycle_ms']:>12.2f} {optimized['max_time_per_cycle_ms']:>12.2f}")
    print(f"  {'Total time (s)':<25} {baseline['total_time_s']:>12.3f} {optimized['total_time_s']:>12.3f}")
    print()
    
    # Function comparison (top 10)
    print("TOP 10 FUNCTIONS BY CUMULATIVE TIME:")
    print()
    print(f"  BASELINE:")
    for i, func in enumerate(baseline.get("top_functions", [])[:10], 1):
        print(f"    {i:>2}. {func.get('cumtime', 0):>8.3f}s  {func.get('ncalls', 0):>8} calls  {func.get('location', 'unknown')}")
    
    print()
    print(f"  OPTIMIZED:")
    for i, func in enumerate(optimized.get("top_functions", [])[:10], 1):
        print(f"    {i:>2}. {func.get('cumtime', 0):>8.3f}s  {func.get('ncalls', 0):>8} calls  {func.get('location', 'unknown')}")
    
    print()
    print("=" * 70)
    if speedup > 0:
        print(f"RESULT: {speedup:.1f}% SPEEDUP achieved")
    else:
        print(f"RESULT: {abs(speedup):.1f}% SLOWDOWN detected")
    print("=" * 70)
    
    # Save comparison report
    comparison_path = PERF_RESULTS_DIR / "comparison.txt"
    with open(comparison_path, 'w', encoding='utf-8') as f:
        f.write(f"Benchmark Comparison: {baseline_tag} vs {optimized_tag}\n")
        f.write(f"{'=' * 70}\n\n")
        f.write(f"Baseline avg cycle time: {baseline_avg:.2f}ms\n")
        f.write(f"Optimized avg cycle time: {optimized_avg:.2f}ms\n")
        f.write(f"Speedup: {speedup:.1f}%\n")
    
    print(f"\nComparison saved to: {comparison_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CycleRunner Performance Profiling Harness"
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="baseline",
        help="Tag for this benchmark run (e.g., 'baseline', 'optimized')"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up cycles (default: 10)"
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=50,
        help="Number of measured cycles (default: 50)"
    )
    parser.add_argument(
        "--slice",
        type=str,
        default="slice_medium",
        help="Curriculum slice name (default: slice_medium)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "rfl"],
        default="baseline",
        help="Execution mode (default: baseline)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare baseline vs optimized results"
    )
    parser.add_argument(
        "--compare-tags",
        type=str,
        nargs=2,
        metavar=("BASELINE", "OPTIMIZED"),
        help="Compare two specific tags"
    )
    parser.add_argument(
        "--use-optimized",
        action="store_true",
        default=True,
        help="Enable MATHLEDGER_PERF_OPT (default: True)"
    )
    parser.add_argument(
        "--no-optimized",
        action="store_true",
        help="Disable MATHLEDGER_PERF_OPT for baseline measurement"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        return compare_benchmarks()
    elif args.compare_tags:
        return compare_benchmarks(args.compare_tags[0], args.compare_tags[1])
    else:
        # Determine optimization state
        use_optimized = args.use_optimized and not args.no_optimized
        
        run_benchmark(
            tag=args.tag,
            warmup_cycles=args.warmup,
            measured_cycles=args.cycles,
            slice_name=args.slice,
            mode=args.mode,
            use_optimized=use_optimized,
        )
        return 0


if __name__ == "__main__":
    sys.exit(main())
