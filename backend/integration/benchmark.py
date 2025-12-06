"""
from backend.repro.determinism import deterministic_unix_timestamp

_GLOBAL_SEED = 0

Integration benchmarking and latency measurement tools.
"""

import time
import statistics
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.integration.bridge import IntegrationBridge
from backend.integration.metrics import IntegrationMetrics


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    operation: str
    iterations: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    success_rate: float
    throughput_ops_per_sec: float


class IntegrationBenchmark:
    """Benchmark integration performance."""

    def __init__(self, bridge: IntegrationBridge):
        self.bridge = bridge
        self.results: List[BenchmarkResult] = []

    def benchmark_operation(
        self,
        operation_name: str,
        operation_fn: Callable,
        iterations: int = 100,
        warmup: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark a single operation.
        
        Args:
            operation_name: Name of the operation
            operation_fn: Function to benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            
        Returns:
            BenchmarkResult with statistics
        """
        for _ in range(warmup):
            try:
                operation_fn()
            except Exception:
                pass

        durations = []
        successes = 0

        start_time = deterministic_unix_timestamp(_GLOBAL_SEED)

        for _ in range(iterations):
            op_start = deterministic_unix_timestamp(_GLOBAL_SEED)
            success = False

            try:
                operation_fn()
                success = True
                successes += 1
            except Exception:
                pass

            op_end = deterministic_unix_timestamp(_GLOBAL_SEED)
            durations.append((op_end - op_start) * 1000)

        end_time = deterministic_unix_timestamp(_GLOBAL_SEED)
        total_duration = end_time - start_time

        durations_sorted = sorted(durations)

        def percentile(data: List[float], p: float) -> float:
            k = (len(data) - 1) * p
            f = int(k)
            c = f + 1
            if c >= len(data):
                return data[-1]
            return data[f] + (k - f) * (data[c] - data[f])

        result = BenchmarkResult(
            operation=operation_name,
            iterations=iterations,
            mean_ms=statistics.mean(durations),
            median_ms=statistics.median(durations),
            min_ms=min(durations),
            max_ms=max(durations),
            p95_ms=percentile(durations_sorted, 0.95),
            p99_ms=percentile(durations_sorted, 0.99),
            success_rate=(successes / iterations) * 100.0,
            throughput_ops_per_sec=iterations / total_duration if total_duration > 0 else 0.0
        )

        self.results.append(result)
        return result

    def benchmark_db_query(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark database query performance."""
        def query_op():
            self.bridge.query_statements(system="pl", limit=10)

        return self.benchmark_operation("db_query", query_op, iterations)

    def benchmark_metrics_fetch(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark metrics fetch performance."""
        def metrics_op():
            self.bridge.get_metrics_summary()

        return self.benchmark_operation("metrics_fetch", metrics_op, iterations)

    def benchmark_redis_enqueue(self, iterations: int = 100) -> BenchmarkResult:
        """Benchmark Redis enqueue performance."""
        def enqueue_op():
            self.bridge.enqueue_verification_job("p -> p", "Propositional")

        return self.benchmark_operation("redis_enqueue", enqueue_op, iterations)

    def benchmark_derivation(self, iterations: int = 10) -> BenchmarkResult:
        """Benchmark derivation performance."""
        def derivation_op():
            self.bridge.execute_derivation(system="pl", steps=5, max_breadth=50)

        return self.benchmark_operation("derivation", derivation_op, iterations)

    def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Run full integration benchmark suite."""
        benchmarks = {
            "db_query": lambda: self.benchmark_db_query(100),
            "metrics_fetch": lambda: self.benchmark_metrics_fetch(100),
            "redis_enqueue": lambda: self.benchmark_redis_enqueue(100),
            "derivation": lambda: self.benchmark_derivation(10)
        }

        results = {}
        for name, bench_fn in benchmarks.items():
            try:
                result = bench_fn()
                results[name] = result
                print(f"[BENCHMARK] {name}: {result.mean_ms:.2f}ms avg, {result.p95_ms:.2f}ms p95")
            except Exception as e:
                print(f"[BENCHMARK] {name}: FAILED - {e}")

        return results

    def check_latency_target(self, target_ms: float = 200.0) -> Dict[str, Any]:
        """
        Check if latency targets are met.
        
        Args:
            target_ms: Target latency in milliseconds
            
        Returns:
            Dictionary with pass/fail status and details
        """
        report = {
            "target_ms": target_ms,
            "passed": True,
            "failures": [],
            "results": {}
        }

        for result in self.results:
            meets_target = result.p95_ms < target_ms
            report["results"][result.operation] = {
                "p95_ms": result.p95_ms,
                "meets_target": meets_target
            }

            if not meets_target:
                report["passed"] = False
                report["failures"].append({
                    "operation": result.operation,
                    "p95_ms": result.p95_ms,
                    "target_ms": target_ms,
                    "delta_ms": result.p95_ms - target_ms
                })

        return report


def run_integration_benchmark(
    db_url: Optional[str] = None,
    redis_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete integration benchmark and return report.
    
    Args:
        db_url: Database URL (optional)
        redis_url: Redis URL (optional)
        
    Returns:
        Comprehensive benchmark report
    """
    bridge = IntegrationBridge(db_url=db_url, redis_url=redis_url)
    benchmark = IntegrationBenchmark(bridge)

    print("[BENCHMARK] Starting integration benchmark suite...")

    results = benchmark.run_full_benchmark()
    target_check = benchmark.check_latency_target(200.0)

    report = {
        "timestamp": deterministic_unix_timestamp(_GLOBAL_SEED),
        "benchmarks": {
            name: {
                "mean_ms": result.mean_ms,
                "median_ms": result.median_ms,
                "p95_ms": result.p95_ms,
                "p99_ms": result.p99_ms,
                "success_rate": result.success_rate,
                "throughput_ops_per_sec": result.throughput_ops_per_sec
            }
            for name, result in results.items()
        },
        "latency_target": target_check
    }

    bridge.close()

    return report
