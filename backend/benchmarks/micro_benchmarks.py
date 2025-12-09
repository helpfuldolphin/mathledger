"""
Micro-Benchmarks for Hash Algorithms

Measures raw performance of individual hash operations across different algorithms.

Author: Manus-H
"""

import hashlib
import time
from dataclasses import dataclass
from typing import List, Callable
import statistics

from basis.crypto.hash_registry import get_hash_function


@dataclass
class MicroBenchmarkResult:
    """
    Result of a micro-benchmark run.
    
    Attributes:
        algorithm_name: Name of hash algorithm tested
        input_size_bytes: Size of input data in bytes
        iterations: Number of iterations performed
        total_time_seconds: Total time for all iterations
        mean_time_ms: Mean time per operation in milliseconds
        median_time_ms: Median time per operation in milliseconds
        p90_time_ms: 90th percentile time in milliseconds
        p95_time_ms: 95th percentile time in milliseconds
        p99_time_ms: 99th percentile time in milliseconds
        throughput_mbps: Throughput in megabytes per second
    """
    
    algorithm_name: str
    input_size_bytes: int
    iterations: int
    total_time_seconds: float
    mean_time_ms: float
    median_time_ms: float
    p90_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    throughput_mbps: float


def benchmark_single_hash(
    hash_func: Callable[[bytes], bytes],
    input_data: bytes,
    iterations: int = 10000,
) -> List[float]:
    """
    Benchmark a single hash function with given input.
    
    Args:
        hash_func: The hash function to benchmark
        input_data: Input data to hash
        iterations: Number of iterations to perform
        
    Returns:
        List of individual operation times in seconds
    """
    times = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        _ = hash_func(input_data)
        end = time.perf_counter()
        times.append(end - start)
    
    return times


def analyze_benchmark_times(
    times: List[float],
    algorithm_name: str,
    input_size: int,
) -> MicroBenchmarkResult:
    """
    Analyze benchmark times and compute statistics.
    
    Args:
        times: List of individual operation times in seconds
        algorithm_name: Name of the algorithm
        input_size: Size of input data in bytes
        
    Returns:
        MicroBenchmarkResult with computed statistics
    """
    total_time = sum(times)
    iterations = len(times)
    
    # Convert to milliseconds for readability
    times_ms = [t * 1000 for t in times]
    
    mean_ms = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    
    # Compute percentiles
    sorted_times_ms = sorted(times_ms)
    p90_idx = int(len(sorted_times_ms) * 0.90)
    p95_idx = int(len(sorted_times_ms) * 0.95)
    p99_idx = int(len(sorted_times_ms) * 0.99)
    
    p90_ms = sorted_times_ms[p90_idx]
    p95_ms = sorted_times_ms[p95_idx]
    p99_ms = sorted_times_ms[p99_idx]
    
    # Compute throughput (MB/s)
    total_bytes = input_size * iterations
    throughput_mbps = (total_bytes / (1024 * 1024)) / total_time
    
    return MicroBenchmarkResult(
        algorithm_name=algorithm_name,
        input_size_bytes=input_size,
        iterations=iterations,
        total_time_seconds=total_time,
        mean_time_ms=mean_ms,
        median_time_ms=median_ms,
        p90_time_ms=p90_ms,
        p95_time_ms=p95_ms,
        p99_time_ms=p99_ms,
        throughput_mbps=throughput_mbps,
    )


def benchmark_sha256(input_size: int, iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark SHA-256."""
    input_data = b"x" * input_size
    
    def sha256_hash(data: bytes) -> bytes:
        return hashlib.sha256(data).digest()
    
    times = benchmark_single_hash(sha256_hash, input_data, iterations)
    return analyze_benchmark_times(times, "SHA-256", input_size)


def benchmark_sha3_256(input_size: int, iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark SHA3-256."""
    input_data = b"x" * input_size
    
    def sha3_256_hash(data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()
    
    times = benchmark_single_hash(sha3_256_hash, input_data, iterations)
    return analyze_benchmark_times(times, "SHA3-256", input_size)


def benchmark_blake3(input_size: int, iterations: int = 10000) -> MicroBenchmarkResult:
    """Benchmark BLAKE3 (placeholder - requires blake3 package)."""
    input_data = b"x" * input_size
    
    try:
        import blake3
        
        def blake3_hash(data: bytes) -> bytes:
            return blake3.blake3(data).digest()
        
        times = benchmark_single_hash(blake3_hash, input_data, iterations)
        return analyze_benchmark_times(times, "BLAKE3", input_size)
    except ImportError:
        # BLAKE3 not available, return placeholder result
        return MicroBenchmarkResult(
            algorithm_name="BLAKE3",
            input_size_bytes=input_size,
            iterations=0,
            total_time_seconds=0.0,
            mean_time_ms=0.0,
            median_time_ms=0.0,
            p90_time_ms=0.0,
            p95_time_ms=0.0,
            p99_time_ms=0.0,
            throughput_mbps=0.0,
        )


def run_micro_benchmarks() -> List[MicroBenchmarkResult]:
    """
    Run comprehensive micro-benchmarks for all hash algorithms.
    
    Tests multiple input sizes: 32B, 64B, 128B, 256B, 512B, 1KB, 4KB, 16KB, 64KB.
    
    Returns:
        List of MicroBenchmarkResult objects
    """
    input_sizes = [32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
    results = []
    
    print("Running micro-benchmarks...")
    print("=" * 80)
    
    for size in input_sizes:
        print(f"\nInput size: {size} bytes")
        
        # Benchmark SHA-256
        sha256_result = benchmark_sha256(size)
        results.append(sha256_result)
        print(f"  SHA-256: {sha256_result.mean_time_ms:.3f} ms (mean), {sha256_result.throughput_mbps:.2f} MB/s")
        
        # Benchmark SHA3-256
        sha3_result = benchmark_sha3_256(size)
        results.append(sha3_result)
        print(f"  SHA3-256: {sha3_result.mean_time_ms:.3f} ms (mean), {sha3_result.throughput_mbps:.2f} MB/s")
        
        # Benchmark BLAKE3
        blake3_result = benchmark_blake3(size)
        if blake3_result.iterations > 0:
            results.append(blake3_result)
            print(f"  BLAKE3: {blake3_result.mean_time_ms:.3f} ms (mean), {blake3_result.throughput_mbps:.2f} MB/s")
    
    print("\n" + "=" * 80)
    print("Micro-benchmarks complete")
    
    return results


def compare_algorithms(results: List[MicroBenchmarkResult]) -> None:
    """
    Compare algorithm performance and print summary.
    
    Args:
        results: List of benchmark results to compare
    """
    print("\n" + "=" * 80)
    print("Algorithm Comparison Summary")
    print("=" * 80)
    
    # Group results by input size
    results_by_size = {}
    for result in results:
        size = result.input_size_bytes
        if size not in results_by_size:
            results_by_size[size] = []
        results_by_size[size].append(result)
    
    # Compare for each input size
    for size in sorted(results_by_size.keys()):
        size_results = results_by_size[size]
        
        print(f"\nInput size: {size} bytes")
        print(f"{'Algorithm':<15} {'Mean (ms)':<12} {'P95 (ms)':<12} {'Throughput (MB/s)':<20}")
        print("-" * 80)
        
        for result in size_results:
            print(
                f"{result.algorithm_name:<15} "
                f"{result.mean_time_ms:<12.3f} "
                f"{result.p95_time_ms:<12.3f} "
                f"{result.throughput_mbps:<20.2f}"
            )
        
        # Find fastest algorithm
        if size_results:
            fastest = min(size_results, key=lambda r: r.mean_time_ms)
            print(f"\nFastest: {fastest.algorithm_name}")
            
            # Compute relative performance
            for result in size_results:
                if result.algorithm_name != fastest.algorithm_name:
                    slowdown = result.mean_time_ms / fastest.mean_time_ms
                    print(f"  {result.algorithm_name} is {slowdown:.2f}x slower than {fastest.algorithm_name}")


def export_results_csv(results: List[MicroBenchmarkResult], filename: str) -> None:
    """
    Export benchmark results to CSV file.
    
    Args:
        results: List of benchmark results
        filename: Output CSV filename
    """
    import csv
    
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Algorithm",
            "Input Size (bytes)",
            "Iterations",
            "Total Time (s)",
            "Mean (ms)",
            "Median (ms)",
            "P90 (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Throughput (MB/s)",
        ])
        
        # Write results
        for result in results:
            writer.writerow([
                result.algorithm_name,
                result.input_size_bytes,
                result.iterations,
                result.total_time_seconds,
                result.mean_time_ms,
                result.median_time_ms,
                result.p90_time_ms,
                result.p95_time_ms,
                result.p99_time_ms,
                result.throughput_mbps,
            ])
    
    print(f"\nResults exported to {filename}")


if __name__ == "__main__":
    # Run benchmarks
    results = run_micro_benchmarks()
    
    # Compare algorithms
    compare_algorithms(results)
    
    # Export results
    export_results_csv(results, "micro_benchmark_results.csv")
