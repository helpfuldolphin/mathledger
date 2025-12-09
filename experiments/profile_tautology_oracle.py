#!/usr/bin/env python3
"""
Phase II Benchmark: Truth-Table Oracle Profiling

Measures baseline performance of the tautology oracle across various formula complexities.
Run before and after optimization to quantify improvements.

Usage:
    python experiments/profile_tautology_oracle.py [--iterations N] [--output FILE]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from normalization.truthtab import is_tautology as truthtab_is_tautology
from normalization.taut import truth_table_is_tautology as taut_is_tautology


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    formula: str
    name: str
    iterations: int
    total_time_ms: float
    mean_time_us: float
    std_dev_us: float
    min_time_us: float
    max_time_us: float
    result: bool


def benchmark_oracle(
    oracle: Callable[[str], bool],
    formula: str,
    name: str,
    iterations: int = 1000
) -> BenchmarkResult:
    """Benchmark a single formula with the given oracle."""
    times_ns: List[float] = []
    result = None
    
    # Warm-up run
    result = oracle(formula)
    
    # Timed runs
    for _ in range(iterations):
        start = time.perf_counter_ns()
        r = oracle(formula)
        end = time.perf_counter_ns()
        times_ns.append(end - start)
        if result is None:
            result = r
    
    times_us = [t / 1000 for t in times_ns]
    
    return BenchmarkResult(
        formula=formula,
        name=name,
        iterations=iterations,
        total_time_ms=sum(times_ns) / 1_000_000,
        mean_time_us=statistics.mean(times_us),
        std_dev_us=statistics.stdev(times_us) if len(times_us) > 1 else 0.0,
        min_time_us=min(times_us),
        max_time_us=max(times_us),
        result=result if result is not None else False,
    )


# Benchmark formulas organized by complexity
BENCHMARK_FORMULAS: List[Tuple[str, str, bool]] = [
    # (formula, name, expected_result)
    
    # Simple formulas (1-2 atoms)
    ("p -> p", "identity", True),
    ("p \\/ ~p", "excluded_middle", True),
    ("p /\\ ~p", "contradiction", False),
    
    # Medium formulas (2-3 atoms)
    ("((p -> q) -> p) -> p", "peirce_law", True),
    ("(p -> q) \\/ (q -> p)", "connectivity", True),
    ("(p /\\ q) -> p", "simplification", True),
    ("p -> (q -> p)", "affirmation_consequent", True),
    ("(p -> q) /\\ (q -> p)", "biconditional_parts", False),
    
    # Complex formulas (3 atoms)
    ("(p -> q) -> ((q -> r) -> (p -> r))", "hypothetical_syllogism", True),
    ("p -> (q -> (p /\\ q))", "conjunction_intro", True),
    ("((p /\\ q) /\\ r) -> p", "nested_simplification", True),
    ("(p \\/ q) -> (q \\/ p)", "disjunction_commutativity", True),
    
    # Non-tautologies for comparison
    ("p -> q", "simple_implication", False),
    ("p /\\ q /\\ r", "triple_conjunction", False),
]


def run_benchmarks(
    iterations: int = 1000,
    output_file: str | None = None
) -> List[BenchmarkResult]:
    """Run all benchmarks and optionally save results."""
    
    results: List[BenchmarkResult] = []
    
    # Header
    header = f"""
================================================================================
TRUTH-TABLE ORACLE BENCHMARK
================================================================================
Iterations per formula: {iterations}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
Python version: {sys.version.split()[0]}
================================================================================
"""
    print(header)
    
    # Test both implementations
    oracles = [
        ("truthtab.is_tautology", truthtab_is_tautology),
        ("taut.truth_table_is_tautology", taut_is_tautology),
    ]
    
    for oracle_name, oracle_fn in oracles:
        print(f"\n--- Oracle: {oracle_name} ---\n")
        print(f"{'Formula':<45} {'Name':<25} {'Mean (μs)':<12} {'Std (μs)':<12} {'Result':<8}")
        print("-" * 110)
        
        for formula, name, expected in BENCHMARK_FORMULAS:
            try:
                result = benchmark_oracle(oracle_fn, formula, name, iterations)
                results.append(result)
                
                # Verify correctness
                status = "✓" if result.result == expected else "✗ MISMATCH"
                
                print(f"{formula:<45} {name:<25} {result.mean_time_us:<12.2f} {result.std_dev_us:<12.2f} {status:<8}")
            except Exception as e:
                print(f"{formula:<45} {name:<25} {'ERROR':<12} {str(e)[:30]}")
    
    # Summary statistics
    summary = "\n" + "=" * 110 + "\n"
    summary += "SUMMARY\n"
    summary += "=" * 110 + "\n"
    
    for oracle_name, _ in oracles:
        oracle_results = [r for r in results if oracle_name.split('.')[0] in str(r)]
        if oracle_results:
            total_mean = sum(r.mean_time_us for r in oracle_results) / len(oracle_results)
            summary += f"\n{oracle_name}:\n"
            summary += f"  Average mean time: {total_mean:.2f} μs\n"
    
    print(summary)
    
    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(header)
            f.write("\n")
            
            for oracle_name, oracle_fn in oracles:
                f.write(f"\n--- Oracle: {oracle_name} ---\n\n")
                f.write(f"{'Formula':<45} {'Name':<25} {'Mean (μs)':<12} {'Std (μs)':<12} {'Min (μs)':<12} {'Max (μs)':<12} {'Result':<8}\n")
                f.write("-" * 130 + "\n")
                
                for formula, name, expected in BENCHMARK_FORMULAS:
                    matching = [r for r in results if r.formula == formula and r.name == name]
                    if matching:
                        r = matching[-1]  # Get the result for this oracle
                        status = "PASS" if r.result == expected else "FAIL"
                        f.write(f"{formula:<45} {name:<25} {r.mean_time_us:<12.2f} {r.std_dev_us:<12.2f} {r.min_time_us:<12.2f} {r.max_time_us:<12.2f} {status:<8}\n")
            
            f.write(summary)
        
        print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile truth-table oracle performance")
    parser.add_argument("--iterations", "-n", type=int, default=1000,
                       help="Number of iterations per formula (default: 1000)")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file path (default: print to stdout only)")
    args = parser.parse_args()
    
    run_benchmarks(
        iterations=args.iterations,
        output_file=args.output
    )


if __name__ == "__main__":
    main()

