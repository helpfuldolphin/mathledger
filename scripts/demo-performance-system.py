#!/usr/bin/env python3
"""
Performance Monitoring System Demo
Cursor B - Performance & Memory Sanity Cartographer

Demonstrates the complete performance monitoring system with all components.
"""

import json
import os
import sys
import time
import subprocess
from datetime import datetime


def run_command(cmd: str, cwd: str = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=60
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 60 seconds"


def demo_performance_system():
    """Demonstrate the complete performance monitoring system."""
    print("="*60)
    print("PERFORMANCE MONITORING SYSTEM DEMO")
    print("Cursor B - Performance & Memory Sanity Cartographer")
    print("="*60)

    # Check if API is running
    print("\n1. Checking API availability...")
    returncode, stdout, stderr = run_command("curl -f http://localhost:8000/health")
    if returncode != 0:
        print("ERROR: API is not running. Please start it with 'make api'")
        return False
    print("PASS: API is running")

    # Create baseline passport
    print("\n2. Creating baseline performance passport...")
    returncode, stdout, stderr = run_command("python scripts/create-baseline-passport.py --api-url http://localhost:8000 --ci")
    if returncode != 0:
        print(f"ERROR: Failed to create baseline passport: {stderr}")
        return False
    print("PASS: Baseline passport created")

    # Validate performance requirements
    print("\n3. Validating performance requirements...")
    returncode, stdout, stderr = run_command("python scripts/validate-performance-requirements.py --api-url http://localhost:8000 --iterations 5 --ci")
    if returncode != 0:
        print(f"ERROR: Performance requirements not met: {stderr}")
        return False
    print("PASS: Performance requirements met")

    # Generate performance passport
    print("\n4. Generating performance passport...")
    returncode, stdout, stderr = run_command("python scripts/generate-performance-passport.py --api-url http://localhost:8000 --baseline baseline_performance_passport.json --ci")
    if returncode != 0:
        print(f"ERROR: Failed to generate performance passport: {stderr}")
        return False
    print("PASS: Performance passport generated")

    # Run CI performance check
    print("\n5. Running CI performance check...")
    returncode, stdout, stderr = run_command("python scripts/ci-performance-check.py --api-url http://localhost:8000 --baseline baseline_performance_passport.json --ci")
    if returncode != 0:
        print(f"ERROR: CI performance check failed: {stderr}")
        return False
    print("PASS: CI performance check completed")

    # Benchmark performance formatter
    print("\n6. Benchmarking performance formatter...")
    returncode, stdout, stderr = run_command("python scripts/performance-formatter.py --benchmark 50")
    if returncode != 0:
        print(f"ERROR: Performance formatter benchmark failed: {stderr}")
        return False
    print("PASS: Performance formatter benchmark completed")

    # Display results
    print("\n7. Performance monitoring results:")
    print("-" * 40)

    # Show baseline passport summary
    if os.path.exists("baseline_performance_passport.json"):
        with open("baseline_performance_passport.json", "r") as f:
            baseline = json.load(f)
        print(f"Baseline Passport:")
        print(f"  Run ID: {baseline.get('run_id', 'N/A')}")
        print(f"  Tests: {baseline['summary']['passed_tests']}/{baseline['summary']['total_tests']} passed")
        print(f"  Max Latency: {baseline['summary']['max_latency_ms']:.3f}ms")
        print(f"  Max Memory: {baseline['summary']['max_memory_mb']:.2f}MB")
        print(f"  Status: {baseline['summary']['overall_status']}")

    # Show current passport summary
    if os.path.exists("performance_passport.json"):
        with open("performance_passport.json", "r") as f:
            current = json.load(f)
        print(f"\nCurrent Passport:")
        print(f"  Run ID: {current.get('run_id', 'N/A')}")
        print(f"  Tests: {current['summary']['passed_tests']}/{current['summary']['total_tests']} passed")
        print(f"  Max Latency: {current['summary']['max_latency_ms']:.3f}ms")
        print(f"  Max Memory: {current['summary']['max_memory_mb']:.2f}MB")
        print(f"  Status: {current['summary']['overall_status']}")

        # Check for regressions
        if current['summary']['performance_regressions'] > 0:
            print(f"  WARNING: {current['summary']['performance_regressions']} performance regressions detected")
        if current['summary']['memory_regressions'] > 0:
            print(f"  WARNING: {current['summary']['memory_regressions']} memory regressions detected")

    # Show validation results
    if os.path.exists("performance_validation_results.json"):
        with open("performance_validation_results.json", "r") as f:
            validation = json.load(f)
        print(f"\nValidation Results:")
        for result in validation:
            print(f"  {result['endpoint']}: {result['overall_status']}")
            print(f"    Latency: avg={result['statistics']['latency_ms']['average']:.3f}ms, max={result['statistics']['latency_ms']['maximum']:.3f}ms")
            print(f"    Memory: avg={result['statistics']['memory_delta_mb']['average']:.3f}MB, max={result['statistics']['memory_delta_mb']['maximum']:.3f}MB")
            print(f"    Objects: avg={result['statistics']['object_delta']['average']:.1f}, max={result['statistics']['object_delta']['maximum']}")

    # Show benchmark results
    if os.path.exists("performance_formatter_benchmark.json"):
        with open("performance_formatter_benchmark.json", "r") as f:
            benchmark = json.load(f)
        print(f"\nBenchmark Results:")
        print(f"  Iterations: {benchmark['iterations']}")
        print(f"  Average Latency: {benchmark['average_latency_ms']:.6f}ms")
        print(f"  Average Memory: {benchmark['average_memory_mb']:.6f}MB")
        print(f"  Average Objects: {benchmark['average_objects']:.1f}")
        print(f"  Max Latency: {benchmark['max_latency_ms']:.6f}ms")
        print(f"  Max Memory: {benchmark['max_memory_mb']:.6f}MB")
        print(f"  Max Objects: {benchmark['max_objects']}")
        print(f"  Total Warnings: {benchmark['total_warnings']}")

    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("All performance monitoring components are working correctly!")
    print("="*60)

    return True


def main():
    """Main function for the demo."""
    print("Starting Performance Monitoring System Demo...")
    print("This demo will test all components of the performance monitoring system.")
    print("Make sure the API server is running with 'make api' before starting.")
    print()

    # Check if we're in the right directory
    if not os.path.exists("scripts/generate-performance-passport.py"):
        print("ERROR: Please run this script from the project root directory")
        sys.exit(1)

    # Run the demo
    success = demo_performance_system()

    if success:
        print("\nSUCCESS: Performance monitoring system demo completed successfully!")
        sys.exit(0)
    else:
        print("\nFAIL: Performance monitoring system demo failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
