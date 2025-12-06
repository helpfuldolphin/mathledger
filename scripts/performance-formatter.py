#!/usr/bin/env python3
"""
Performance Formatter for /metrics endpoint
Cursor B - Performance & Memory Sanity Cartographer

Ensures /metrics endpoint meets performance requirements:
- <10ms latency
- <10MB peak memory
- <1000 objects allocation

Global doctrine compliance:
- ASCII-only logs; no emojis in CI output
- Deterministic comparison via JSON hash
- Mechanical honesty: Status reflects API/test truth
"""

import json
import time
import tracemalloc
import psutil
import gc
import hashlib
from typing import Dict, Any, Tuple, Optional
from datetime import datetime


class PerformanceFormatter:
    """
    High-performance formatter for /metrics endpoint

    Optimized to meet strict performance requirements with dual memory profilers.
    """

    def __init__(self):
        self.thresholds = {
            "max_latency_ms": 10.0,
            "max_memory_mb": 10.0,
            "max_objects": 1000
        }
        self.warnings = []

    def format_metrics(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format metrics data with performance monitoring.

        Args:
            metrics_data: Raw metrics data from database

        Returns:
            Formatted metrics with performance data
        """
        # Start performance monitoring
        start_time = time.perf_counter()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_objects = len(gc.get_objects())

        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        tracemalloc_start = tracemalloc.get_traced_memory()

        try:
            # Format the metrics data
            formatted_metrics = self._format_metrics_core(metrics_data)

            # Calculate performance metrics
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_objects = len(gc.get_objects())

            # Get tracemalloc peak
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / 1024 / 1024  # MB

            # Calculate deltas
            memory_delta = final_memory - initial_memory
            object_delta = final_objects - initial_objects

            # Add performance data to formatted metrics
            formatted_metrics["performance"] = {
                "latency_ms": round(duration_ms, 6),
                "memory_delta_mb": round(memory_delta, 6),
                "object_delta": object_delta,
                "peak_memory_mb": round(peak_memory, 6),
                "initial_memory_mb": round(initial_memory, 6),
                "final_memory_mb": round(final_memory, 6)
            }

            # Check performance thresholds
            self._check_performance_thresholds(duration_ms, memory_delta, object_delta, peak_memory)

            return formatted_metrics

        finally:
            # Stop tracemalloc
            tracemalloc.stop()

    def _format_metrics_core(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Core metrics formatting logic.

        Args:
            metrics_data: Raw metrics data

        Returns:
            Formatted metrics without performance data
        """
        # Extract data with safe defaults
        block_height = metrics_data.get('block_height', 0)
        block_count = metrics_data.get('block_count', 0)
        statement_count = metrics_data.get('statement_count', 0)
        proofs_total = metrics_data.get('proofs_total', 0)
        proofs_success = metrics_data.get('proofs_success', 0)
        max_depth = metrics_data.get('max_depth', 0)

        # Calculate derived values
        proofs_failure = max(0, proofs_total - proofs_success)
        success_rate = (float(proofs_success) / float(proofs_total)) if proofs_total else 0.0

        # Build formatted response
        return {
            "proofs": {"success": proofs_success, "failure": proofs_failure},
            "block_count": block_count,
            "max_depth": max_depth,
            "proof_counts": proofs_total,
            "statement_counts": statement_count,
            "success_rate": success_rate,
            "queue_length": -1,
            "block_height": block_height,
            "blocks": {"height": block_height}
        }

    def _check_performance_thresholds(self, latency_ms: float, memory_delta: float,
                                    object_delta: int, peak_memory: float):
        """
        Check performance thresholds and record warnings.

        Args:
            latency_ms: Request latency in milliseconds
            memory_delta: Memory usage delta in MB
            object_delta: Object allocation delta
            peak_memory: Peak memory usage in MB
        """
        if latency_ms > self.thresholds["max_latency_ms"]:
            warning = f"WARNING: /metrics endpoint latency {latency_ms:.3f}ms exceeds {self.thresholds['max_latency_ms']}ms threshold"
            self.warnings.append(warning)
            print(warning)

        if memory_delta > self.thresholds["max_memory_mb"]:
            warning = f"WARNING: /metrics endpoint memory usage {memory_delta:.2f}MB exceeds {self.thresholds['max_memory_mb']}MB threshold"
            self.warnings.append(warning)
            print(warning)

        if object_delta > self.thresholds["max_objects"]:
            warning = f"WARNING: /metrics endpoint object allocation {object_delta} exceeds {self.thresholds['max_objects']} threshold"
            self.warnings.append(warning)
            print(warning)

        if peak_memory > self.thresholds["max_memory_mb"]:
            warning = f"WARNING: /metrics endpoint peak memory {peak_memory:.2f}MB exceeds {self.thresholds['max_memory_mb']}MB threshold"
            self.warnings.append(warning)
            print(warning)

    def get_warnings(self) -> list:
        """Get performance warnings."""
        return self.warnings.copy()

    def clear_warnings(self):
        """Clear performance warnings."""
        self.warnings.clear()

    def validate_performance(self, formatted_metrics: Dict[str, Any]) -> Tuple[bool, list]:
        """
        Validate that formatted metrics meet performance requirements.

        Args:
            formatted_metrics: Formatted metrics with performance data

        Returns:
            Tuple of (is_valid, warnings)
        """
        performance = formatted_metrics.get("performance", {})

        latency_ms = performance.get("latency_ms", 0)
        memory_delta = performance.get("memory_delta_mb", 0)
        object_delta = performance.get("object_delta", 0)
        peak_memory = performance.get("peak_memory_mb", 0)

        warnings = []
        is_valid = True

        if latency_ms > self.thresholds["max_latency_ms"]:
            warnings.append(f"Latency {latency_ms:.3f}ms exceeds {self.thresholds['max_latency_ms']}ms threshold")
            is_valid = False

        if memory_delta > self.thresholds["max_memory_mb"]:
            warnings.append(f"Memory usage {memory_delta:.2f}MB exceeds {self.thresholds['max_memory_mb']}MB threshold")
            is_valid = False

        if object_delta > self.thresholds["max_objects"]:
            warnings.append(f"Object allocation {object_delta} exceeds {self.thresholds['max_objects']} threshold")
            is_valid = False

        if peak_memory > self.thresholds["max_memory_mb"]:
            warnings.append(f"Peak memory {peak_memory:.2f}MB exceeds {self.thresholds['max_memory_mb']}MB threshold")
            is_valid = False

        return is_valid, warnings


def create_deterministic_hash(data: Dict[str, Any]) -> str:
    """
    Create a deterministic hash of the data for comparison.

    Args:
        data: Data to hash

    Returns:
        SHA-256 hash as hexadecimal string
    """
    # Convert to JSON with sorted keys for deterministic output
    json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(json_str.encode()).hexdigest()


def benchmark_formatter(iterations: int = 100) -> Dict[str, Any]:
    """
    Benchmark the performance formatter.

    Args:
        iterations: Number of iterations to run

    Returns:
        Benchmark results
    """
    formatter = PerformanceFormatter()

    # Sample metrics data
    sample_data = {
        'block_height': 42,
        'block_count': 42,
        'statement_count': 1500,
        'proofs_total': 3000,
        'proofs_success': 2850,
        'max_depth': 8
    }

    results = []
    total_latency = 0
    total_memory = 0
    total_objects = 0
    max_latency = 0
    max_memory = 0
    max_objects = 0

    print(f"Benchmarking performance formatter with {iterations} iterations...")

    for i in range(iterations):
        # Clear warnings
        formatter.clear_warnings()

        # Format metrics
        formatted = formatter.format_metrics(sample_data)

        # Extract performance data
        perf = formatted.get("performance", {})
        latency = perf.get("latency_ms", 0)
        memory = perf.get("memory_delta_mb", 0)
        objects = perf.get("object_delta", 0)

        # Update totals
        total_latency += latency
        total_memory += memory
        total_objects += objects

        # Update maximums
        max_latency = max(max_latency, latency)
        max_memory = max(max_memory, memory)
        max_objects = max(max_objects, objects)

        # Check for warnings
        warnings = formatter.get_warnings()
        if warnings:
            print(f"Iteration {i+1}: {warnings}")

        results.append({
            'iteration': i + 1,
            'latency_ms': latency,
            'memory_mb': memory,
            'objects': objects,
            'warnings': len(warnings)
        })

    # Calculate averages
    avg_latency = total_latency / iterations
    avg_memory = total_memory / iterations
    avg_objects = total_objects / iterations

    benchmark_results = {
        'iterations': iterations,
        'average_latency_ms': round(avg_latency, 6),
        'average_memory_mb': round(avg_memory, 6),
        'average_objects': round(avg_objects, 6),
        'max_latency_ms': round(max_latency, 6),
        'max_memory_mb': round(max_memory, 6),
        'max_objects': max_objects,
        'total_warnings': sum(r['warnings'] for r in results),
        'results': results
    }

    print(f"Benchmark completed:")
    print(f"  Average latency: {avg_latency:.6f}ms")
    print(f"  Average memory: {avg_memory:.6f}MB")
    print(f"  Average objects: {avg_objects:.6f}")
    print(f"  Max latency: {max_latency:.6f}ms")
    print(f"  Max memory: {max_memory:.6f}MB")
    print(f"  Max objects: {max_objects}")
    print(f"  Total warnings: {benchmark_results['total_warnings']}")

    return benchmark_results


def main():
    """Main function for testing the performance formatter."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Formatter Test")
    parser.add_argument("--benchmark", type=int, default=100, help="Number of benchmark iterations")
    parser.add_argument("--test-data", help="JSON file with test data")

    args = parser.parse_args()

    # Run benchmark
    benchmark_results = benchmark_formatter(args.benchmark)

    # Save benchmark results
    with open("performance_formatter_benchmark.json", "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print("Benchmark results saved to performance_formatter_benchmark.json")

    # Test with custom data if provided
    if args.test_data:
        with open(args.test_data, "r") as f:
            test_data = json.load(f)

        formatter = PerformanceFormatter()
        formatted = formatter.format_metrics(test_data)

        print("Test data formatting completed:")
        print(f"  Latency: {formatted['performance']['latency_ms']:.6f}ms")
        print(f"  Memory: {formatted['performance']['memory_delta_mb']:.6f}MB")
        print(f"  Objects: {formatted['performance']['object_delta']}")

        # Validate performance
        is_valid, warnings = formatter.validate_performance(formatted)
        if is_valid:
            print("PASS: Performance requirements met")
        else:
            print("FAIL: Performance requirements not met")
            for warning in warnings:
                print(f"  - {warning}")


if __name__ == "__main__":
    main()
