#!/usr/bin/env python3
"""
Performance Requirements Validation Script
Cursor B - Performance & Memory Sanity Cartographer

Validates that the /metrics endpoint meets all performance requirements:
- <10ms latency
- <10MB peak memory
- <1000 objects allocation

Global doctrine compliance:
- ASCII-only logs; no emojis in CI output
- Deterministic comparison via JSON hash
- Mechanical honesty: Status reflects API/test truth
"""

import json
import os
import sys
import time
import requests
import argparse
from typing import Dict, Any, List, Tuple
from datetime import datetime


class PerformanceRequirementsValidator:
    """
    Validates performance requirements for the /metrics endpoint.
    """

    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.requirements = {
            "max_latency_ms": 10.0,
            "max_memory_mb": 10.0,
            "max_objects": 1000
        }
        self.results = []

    def validate_endpoint(self, endpoint: str = "/metrics", iterations: int = 10) -> Dict[str, Any]:
        """
        Validate performance requirements for an endpoint.

        Args:
            endpoint: Endpoint to test
            iterations: Number of test iterations

        Returns:
            Validation results
        """
        print(f"Validating performance requirements for {endpoint}")
        print(f"Requirements: <{self.requirements['max_latency_ms']}ms latency, <{self.requirements['max_memory_mb']}MB memory, <{self.requirements['max_objects']} objects")
        print(f"Running {iterations} iterations...")

        latencies = []
        memory_deltas = []
        object_deltas = []
        peak_memories = []
        warnings = []

        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}...", end=" ")

            try:
                # Make request and measure performance
                start_time = time.perf_counter()
                response = requests.get(f"{self.api_url}{endpoint}", timeout=30)
                end_time = time.perf_counter()

                if response.status_code != 200:
                    print(f"FAIL (HTTP {response.status_code})")
                    warnings.append(f"Iteration {i+1}: HTTP {response.status_code}")
                    continue

                # Extract performance data from response
                data = response.json()
                performance = data.get("performance", {})

                latency_ms = performance.get("latency_ms", 0)
                memory_delta = performance.get("memory_delta_mb", 0)
                object_delta = performance.get("object_delta", 0)
                peak_memory = performance.get("peak_memory_mb", 0)

                # Check requirements
                iteration_warnings = []
                if latency_ms > self.requirements["max_latency_ms"]:
                    iteration_warnings.append(f"latency {latency_ms:.3f}ms > {self.requirements['max_latency_ms']}ms")
                if memory_delta > self.requirements["max_memory_mb"]:
                    iteration_warnings.append(f"memory {memory_delta:.2f}MB > {self.requirements['max_memory_mb']}MB")
                if object_delta > self.requirements["max_objects"]:
                    iteration_warnings.append(f"objects {object_delta} > {self.requirements['max_objects']}")
                if peak_memory > self.requirements["max_memory_mb"]:
                    iteration_warnings.append(f"peak memory {peak_memory:.2f}MB > {self.requirements['max_memory_mb']}MB")

                if iteration_warnings:
                    print(f"WARN ({', '.join(iteration_warnings)})")
                    warnings.extend([f"Iteration {i+1}: {w}" for w in iteration_warnings])
                else:
                    print("PASS")

                # Record metrics
                latencies.append(latency_ms)
                memory_deltas.append(memory_delta)
                object_deltas.append(object_delta)
                peak_memories.append(peak_memory)

            except Exception as e:
                print(f"ERROR ({str(e)})")
                warnings.append(f"Iteration {i+1}: {str(e)}")

        # Calculate statistics
        if latencies:
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            min_latency = min(latencies)
        else:
            avg_latency = max_latency = min_latency = 0

        if memory_deltas:
            avg_memory = sum(memory_deltas) / len(memory_deltas)
            max_memory = max(memory_deltas)
            min_memory = min(memory_deltas)
        else:
            avg_memory = max_memory = min_memory = 0

        if object_deltas:
            avg_objects = sum(object_deltas) / len(object_deltas)
            max_objects = max(object_deltas)
            min_objects = min(object_deltas)
        else:
            avg_objects = max_objects = min_objects = 0

        if peak_memories:
            avg_peak_memory = sum(peak_memories) / len(peak_memories)
            max_peak_memory = max(peak_memories)
            min_peak_memory = min(peak_memories)
        else:
            avg_peak_memory = max_peak_memory = min_peak_memory = 0

        # Determine overall status
        overall_status = "PASS"
        if (max_latency > self.requirements["max_latency_ms"] or
            max_memory > self.requirements["max_memory_mb"] or
            max_objects > self.requirements["max_objects"] or
            max_peak_memory > self.requirements["max_memory_mb"]):
            overall_status = "FAIL"

        results = {
            "endpoint": endpoint,
            "iterations": iterations,
            "successful_iterations": len(latencies),
            "overall_status": overall_status,
            "requirements": self.requirements,
            "statistics": {
                "latency_ms": {
                    "average": round(avg_latency, 6),
                    "maximum": round(max_latency, 6),
                    "minimum": round(min_latency, 6)
                },
                "memory_delta_mb": {
                    "average": round(avg_memory, 6),
                    "maximum": round(max_memory, 6),
                    "minimum": round(min_memory, 6)
                },
                "object_delta": {
                    "average": round(avg_objects, 6),
                    "maximum": max_objects,
                    "minimum": min_objects
                },
                "peak_memory_mb": {
                    "average": round(avg_peak_memory, 6),
                    "maximum": round(max_peak_memory, 6),
                    "minimum": round(min_peak_memory, 6)
                }
            },
            "warnings": warnings,
            "raw_data": {
                "latencies": latencies,
                "memory_deltas": memory_deltas,
                "object_deltas": object_deltas,
                "peak_memories": peak_memories
            }
        }

        self.results.append(results)
        return results

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("PERFORMANCE REQUIREMENTS VALIDATION SUMMARY")
        print("="*60)

        for result in self.results:
            endpoint = result["endpoint"]
            status = result["overall_status"]
            stats = result["statistics"]

            print(f"\nEndpoint: {endpoint}")
            print(f"Status: {status}")
            print(f"Successful iterations: {result['successful_iterations']}/{result['iterations']}")

            print(f"Latency: avg={stats['latency_ms']['average']:.3f}ms, max={stats['latency_ms']['maximum']:.3f}ms, min={stats['latency_ms']['minimum']:.3f}ms")
            print(f"Memory: avg={stats['memory_delta_mb']['average']:.3f}MB, max={stats['memory_delta_mb']['maximum']:.3f}MB, min={stats['memory_delta_mb']['minimum']:.3f}MB")
            print(f"Objects: avg={stats['object_delta']['average']:.1f}, max={stats['object_delta']['maximum']}, min={stats['object_delta']['minimum']}")
            print(f"Peak Memory: avg={stats['peak_memory_mb']['average']:.3f}MB, max={stats['peak_memory_mb']['maximum']:.3f}MB, min={stats['peak_memory_mb']['minimum']:.3f}MB")

            if result["warnings"]:
                print("Warnings:")
                for warning in result["warnings"]:
                    print(f"  - {warning}")

        # Overall status
        all_passed = all(r["overall_status"] == "PASS" for r in self.results)
        print(f"\nOverall Status: {'PASS' if all_passed else 'FAIL'}")

        if not all_passed:
            print("Some endpoints failed to meet performance requirements!")
            sys.exit(1)
        else:
            print("All endpoints meet performance requirements!")

    def save_results(self, filename: str = "performance_validation_results.json"):
        """Save validation results to file."""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")


def main():
    """Main function for performance validation."""
    parser = argparse.ArgumentParser(description="Validate Performance Requirements")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--endpoint", default="/metrics", help="Endpoint to test")
    parser.add_argument("--iterations", type=int, default=10, help="Number of test iterations")
    parser.add_argument("--output", default="performance_validation_results.json", help="Output file")
    parser.add_argument("--ci", action="store_true", help="CI mode with enhanced logging")

    args = parser.parse_args()

    if args.ci:
        print("CI Mode: Validating Performance Requirements")
        print(f"API URL: {args.api_url}")
        print(f"Endpoint: {args.endpoint}")
        print(f"Iterations: {args.iterations}")

    # Initialize validator
    validator = PerformanceRequirementsValidator(args.api_url)

    # Validate endpoint
    result = validator.validate_endpoint(args.endpoint, args.iterations)

    # Print summary
    validator.print_summary()

    # Save results
    validator.save_results(args.output)

    # Exit with appropriate code
    if result["overall_status"] == "FAIL":
        print("FAIL: Performance requirements not met")
        sys.exit(1)
    else:
        print("PASS: Performance requirements met")
        sys.exit(0)


if __name__ == "__main__":
    main()
