#!/usr/bin/env python3
"""
Comprehensive Performance Benchmarking Harness for MathLedger DA Pipeline

Profiles and benchmarks all hot paths:
- Modus Ponens rule application (apply_modus_ponens)
- Expression canonicalization (normalize)
- Congruence closure operations (CC.assert_eqs)
- Derivation engine operations

Usage:
    python tools/perf/perf_bench.py --all
    python tools/perf/perf_bench.py --target modus_ponens
    python tools/perf/perf_bench.py --target canonicalization
    python tools/perf/perf_bench.py --target congruence_closure
    python tools/perf/perf_bench.py --baseline  # Generate baseline CSV

Outputs:
    - artifacts/perf/{timestamp}/bench.json
    - artifacts/perf/{timestamp}/report.txt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Set, Tuple

from backend.repro.determinism import deterministic_isoformat, deterministic_slug

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "backend"))


class PerfBench:
    """Performance benchmarking harness for MathLedger DA pipeline."""

    def __init__(self, output_dir: str = None):
        self.results: Dict[str, Any] = {}
        seed = output_dir if output_dir is not None else "perf_bench_default"
        self.run_slug = deterministic_slug("perf_bench", seed, length=16)
        self.timestamp = deterministic_isoformat("perf_bench", seed)
        
        if output_dir is None:
            output_dir = f"artifacts/perf/{self.run_slug}"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def measure_performance(
        self,
        func: Callable,
        *args,
        iterations: int = 10,
        warmup: int = 2,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Measure wall-time and call counts for a function.
        
        Args:
            func: Function to benchmark
            *args: Positional arguments to func
            iterations: Number of measurement iterations
            warmup: Number of warmup iterations (not measured)
            **kwargs: Keyword arguments to func
            
        Returns:
            Dictionary with timing statistics
        """
        for _ in range(warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass

        times = []
        results = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                results.append(result)
            except Exception as e:
                print(f"Error during benchmark: {e}")
                result = None
            end_time = time.perf_counter()
            times.append(end_time - start_time)

        if not times:
            return {
                "error": "No successful iterations",
                "iterations": 0,
                "wall_time_ms": 0,
            }

        return {
            "wall_time_ms": sum(times) * 1000 / len(times),
            "min_time_ms": min(times) * 1000,
            "max_time_ms": max(times) * 1000,
            "median_time_ms": sorted(times)[len(times) // 2] * 1000,
            "std_dev_ms": self._std_dev(times) * 1000,
            "iterations": len(times),
            "warmup_iterations": warmup,
            "result_size": len(results[0]) if results and results[0] is not None else 0,
        }

    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def generate_synthetic_statements(self, size: int) -> Set[str]:
        """Generate synthetic dataset with atoms and implications for MP testing."""
        statements = set()
        atoms_count = size // 2
        
        for i in range(1, atoms_count + 1):
            statements.add(f"p{i}")
            statements.add(f"p{i}->q{i}")
        
        return statements

    def generate_synthetic_formulas(self, count: int) -> List[str]:
        """Generate synthetic formulas for canonicalization testing."""
        formulas = []
        
        for i in range(count // 4):
            formulas.append(f"p{i}->q{i}")
            formulas.append(f"(p{i})->(q{i})")  # With extra parens
        
        for i in range(count // 4):
            formulas.append(f"p{i}/\\q{i}")
            formulas.append(f"q{i}/\\p{i}")  # Commutative variant
        
        for i in range(count // 4):
            formulas.append(f"p{i}\\/q{i}")
            formulas.append(f"q{i}\\/p{i}")  # Commutative variant
        
        for i in range(count // 4):
            formulas.append(f"(p{i}->q{i})->r{i}")
            formulas.append(f"p{i}->(q{i}->r{i})")
        
        return formulas[:count]

    def bench_modus_ponens(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark Modus Ponens rule application."""
        if sizes is None:
            sizes = [100, 500, 1000, 5000, 10000]
        
        print("\n=== Benchmarking Modus Ponens ===")
        
        try:
            from axiom_engine.rules import apply_modus_ponens
        except ImportError:
            print("ERROR: Could not import apply_modus_ponens")
            return {"error": "Import failed"}
        
        results = {}
        
        for size in sizes:
            print(f"\nTesting with {size} statements...")
            statements = self.generate_synthetic_statements(size)
            
            perf = self.measure_performance(
                apply_modus_ponens,
                statements,
                iterations=10 if size <= 1000 else 3,
                warmup=2
            )
            
            perf["input_size"] = size
            results[f"mp_{size}"] = perf
            
            print(f"  Avg: {perf['wall_time_ms']:.4f}ms")
            print(f"  Min: {perf['min_time_ms']:.4f}ms")
            print(f"  Max: {perf['max_time_ms']:.4f}ms")
            print(f"  Derivations: {perf['result_size']}")
        
        return results

    def bench_canonicalization(self, counts: List[int] = None) -> Dict[str, Any]:
        """Benchmark expression canonicalization."""
        if counts is None:
            counts = [100, 500, 1000, 5000, 10000]
        
        print("\n=== Benchmarking Canonicalization ===")
        
        try:
            from logic.canon import normalize
        except ImportError:
            print("ERROR: Could not import normalize")
            return {"error": "Import failed"}
        
        results = {}
        
        for count in counts:
            print(f"\nTesting with {count} formulas...")
            formulas = self.generate_synthetic_formulas(count)
            
            def normalize_batch(formulas):
                return [normalize(f) for f in formulas]
            
            perf = self.measure_performance(
                normalize_batch,
                formulas,
                iterations=10 if count <= 1000 else 3,
                warmup=2
            )
            
            perf["input_size"] = count
            results[f"canon_{count}"] = perf
            
            print(f"  Avg: {perf['wall_time_ms']:.4f}ms")
            print(f"  Min: {perf['min_time_ms']:.4f}ms")
            print(f"  Max: {perf['max_time_ms']:.4f}ms")
            print(f"  Throughput: {count / (perf['wall_time_ms'] / 1000):.0f} formulas/sec")
        
        return results

    def bench_cache_effectiveness(self) -> Dict[str, Any]:
        """Benchmark LRU cache effectiveness for normalization."""
        print("\n=== Benchmarking Cache Effectiveness ===")
        
        try:
            from axiom_engine.rules import _cached_normalize
        except ImportError:
            print("ERROR: Could not import _cached_normalize")
            return {"error": "Import failed"}
        
        _cached_normalize.cache_clear()
        
        test_formulas = [
            "p->q", "(p)->(q)", "p->q",  # Same normalized form
            "r/\\s", "(r)/\\(s)", "r/\\s",  # Same normalized form
            "a\\/b", "a\\/b", "(a\\/b)",  # Same normalized form
        ] * 100  # Repeat to test cache hits
        
        start_time = time.perf_counter()
        results = [_cached_normalize(f) for f in test_formulas]
        end_time = time.perf_counter()
        
        cache_info = _cached_normalize.cache_info()
        
        result = {
            "wall_time_ms": (end_time - start_time) * 1000,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses,
            "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) 
                       if (cache_info.hits + cache_info.misses) > 0 else 0,
            "formulas_processed": len(test_formulas),
            "unique_results": len(set(results)),
            "cache_size": cache_info.currsize,
            "cache_maxsize": cache_info.maxsize,
        }
        
        print(f"\n  Formulas processed: {result['formulas_processed']}")
        print(f"  Cache hits: {result['cache_hits']}")
        print(f"  Cache misses: {result['cache_misses']}")
        print(f"  Hit rate: {result['hit_rate']:.2%}")
        print(f"  Total time: {result['wall_time_ms']:.4f}ms")
        
        return {"cache_effectiveness": result}

    def bench_congruence_closure(self, sizes: List[int] = None) -> Dict[str, Any]:
        """Benchmark congruence closure operations."""
        if sizes is None:
            sizes = [10, 50, 100, 500, 1000]
        
        print("\n=== Benchmarking Congruence Closure ===")
        
        try:
            from fol_eq.cc import CC, const, fun
        except ImportError:
            print("ERROR: Could not import CC")
            return {"error": "Import failed"}
        
        results = {}
        
        for size in sizes:
            print(f"\nTesting with {size} equations...")
            
            eqs = []
            for i in range(size):
                a = const(f"a{i}")
                b = const(f"b{i}")
                eqs.append((a, b))
            
            def run_cc(eqs):
                cc = CC()
                return cc.assert_eqs(eqs)
            
            perf = self.measure_performance(
                run_cc,
                eqs,
                iterations=10 if size <= 100 else 3,
                warmup=2
            )
            
            perf["input_size"] = size
            results[f"cc_{size}"] = perf
            
            print(f"  Avg: {perf['wall_time_ms']:.4f}ms")
            print(f"  Min: {perf['min_time_ms']:.4f}ms")
            print(f"  Max: {perf['max_time_ms']:.4f}ms")
        
        return results

    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks and collect results."""
        print("=" * 60)
        print("MathLedger DA Pipeline Performance Benchmarking")
        print("=" * 60)
        
        all_results = {
            "timestamp": self.timestamp,
            "benchmarks": {}
        }
        
        all_results["benchmarks"]["modus_ponens"] = self.bench_modus_ponens()
        all_results["benchmarks"]["canonicalization"] = self.bench_canonicalization()
        all_results["benchmarks"]["cache"] = self.bench_cache_effectiveness()
        all_results["benchmarks"]["congruence_closure"] = self.bench_congruence_closure()
        
        return all_results

    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to JSON and generate report."""
        json_path = self.output_dir / "bench.json"
        canonical = json.dumps(
            results,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        with open(json_path, "w", encoding="ascii") as f:
            f.write(canonical)
        print(f"\n\nResults saved to: {json_path}")
        
        report_path = self.output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("MathLedger DA Pipeline Performance Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            for suite_name, suite_results in results.get("benchmarks", {}).items():
                f.write(f"\n{suite_name.upper()}\n")
                f.write("-" * 60 + "\n")
                
                if isinstance(suite_results, dict) and "error" not in suite_results:
                    for test_name, test_result in suite_results.items():
                        if isinstance(test_result, dict):
                            f.write(f"\n{test_name}:\n")
                            for key, value in test_result.items():
                                if isinstance(value, float):
                                    f.write(f"  {key}: {value:.4f}\n")
                                else:
                                    f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {suite_results}\n")
        
        print(f"Report saved to: {report_path}")

    def generate_baseline_csv(self):
        """Generate baseline CSV for CI regression detection."""
        import csv
        
        print("\n=== Generating Baseline CSV ===")
        
        mp_results = self.bench_modus_ponens(sizes=[100, 1000, 10000])
        
        csv_path = Path("artifacts/perf/baseline.csv")
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "dataset_size",
                "avg_time_ms",
                "max_time_ms",
                "derivations",
                "timestamp"
            ])
            
            for key, result in sorted(mp_results.items(), key=lambda x: x[1].get("input_size", 0)):
                if "error" not in result:
                    writer.writerow([
                        result.get("input_size", 0),
                        result.get("wall_time_ms", 0),
                        result.get("max_time_ms", 0),
                        result.get("result_size", 0),
                        self.timestamp
                    ])
        
        print(f"Baseline CSV saved to: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="MathLedger DA Pipeline Performance Benchmarking Harness"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--target",
        choices=["modus_ponens", "canonicalization", "congruence_closure", "cache"],
        help="Run specific benchmark target"
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Generate baseline CSV for CI"
    )
    parser.add_argument(
        "--output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    bench = PerfBench(output_dir=args.output)
    
    if args.baseline:
        bench.generate_baseline_csv()
        return 0
    
    if args.all:
        results = bench.run_all_benchmarks()
        bench.save_results(results)
        return 0
    
    if args.target:
        results = {
            "timestamp": bench.timestamp,
            "benchmarks": {}
        }
        
        if args.target == "modus_ponens":
            results["benchmarks"]["modus_ponens"] = bench.bench_modus_ponens()
        elif args.target == "canonicalization":
            results["benchmarks"]["canonicalization"] = bench.bench_canonicalization()
        elif args.target == "congruence_closure":
            results["benchmarks"]["congruence_closure"] = bench.bench_congruence_closure()
        elif args.target == "cache":
            results["benchmarks"]["cache"] = bench.bench_cache_effectiveness()
        
        bench.save_results(results)
        return 0
    
    results = bench.run_all_benchmarks()
    bench.save_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
