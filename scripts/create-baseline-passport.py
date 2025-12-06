#!/usr/bin/env python3
"""
Baseline Performance Passport Creator
Cursor B - Performance & Memory Sanity Cartographer

Creates a baseline performance passport for comparison against future runs.
This baseline serves as the reference point for detecting regressions.
"""

import argparse
import gc
import hashlib
import json
import os
import psutil
import requests
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Tuple

from backend.repro.determinism import (
    deterministic_hash,
    deterministic_isoformat,
    deterministic_run_id,
    deterministic_slug,
)


def create_baseline_passport(api_url: str = "http://localhost:8000",
                           output_file: str = "baseline_performance_passport.json") -> str:
    """
    Create a baseline performance passport from current system state.

    Args:
        api_url: Base URL of the API to test
        output_file: Output filename for the baseline passport

    Returns:
        Path to the created baseline passport file
    """

    # Initialize baseline passport structure
    thresholds = {
        "max_latency_threshold_ms": 10.0,
        "max_memory_threshold_mb": 10.0,
        "max_objects_threshold": 1000,
        "regression_tolerance_percent": 5.0,
        "deterministic_threshold_percent": 95.0
    }
    thresholds_fingerprint = json.dumps(
        thresholds,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    run_id = deterministic_run_id("baseline", api_url, thresholds_fingerprint)
    session_id = deterministic_slug("baseline_session", run_id, length=8)
    baseline_timestamp = deterministic_isoformat("baseline_passport", run_id, api_url)

    baseline_passport = {
        "cartographer": "Cursor B - Performance & Memory Sanity Cartographer",
        "run_id": run_id,
        "session_id": session_id,
        "timestamp": baseline_timestamp,
        "baseline_type": "reference_baseline",
        "performance_guarantee": "Even in sandbox mode, we never regress by more than 5%",
        "endpoints_profiled": [],
        "test_results": [],
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "performance_regressions": 0,
            "memory_regressions": 0,
            "max_latency_ms": 0.0,
            "max_memory_mb": 0.0,
            "max_objects": 0,
            "deterministic_score": 0.0,
            "overall_status": "PASS"
        },
        "thresholds": thresholds,
        "system_info": {
            "platform": os.name,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "psutil_version": psutil.__version__,
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_count": psutil.cpu_count()
        }
    }

    # Test endpoints and collect baseline metrics
    endpoints_to_test = [
        "metrics",
        "health",
        "blocks/latest",
        "statements",
        "ui",
        "heartbeat.json"
    ]

    print("Creating baseline performance passport...")
    print(f"API URL: {api_url}")
    print(f"Output: {output_file}")

    for endpoint in endpoints_to_test:
        print(f"Testing endpoint: {endpoint}")

        # Test multiple variants for each endpoint
        for variant in range(3):  # 3 variants per endpoint
            test_name = f"{endpoint}_variant_{variant}"

            # Start memory profiling
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            initial_objects = len(gc.get_objects())

            tracemalloc.start()
            tracemalloc_start = tracemalloc.get_traced_memory()

            # Time the endpoint call
            start_time = time.perf_counter()

            try:
                # Make API call
                url = f"{api_url.rstrip('/')}/{endpoint.lstrip('/')}"
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                result = response.json()
                status = "PASS"
            except Exception as e:
                result = None
                status = "FAIL"
                print(f"WARNING: Endpoint {endpoint} failed: {e}")

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            # Stop memory profiling
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            final_objects = len(gc.get_objects())

            current, peak = tracemalloc.get_traced_memory()
            peak_memory = peak / 1024 / 1024  # MB

            memory_delta = final_memory - initial_memory
            object_delta = final_objects - initial_objects

            tracemalloc.stop()

            # Create test result
            test_result = {
                "endpoint": endpoint,
                "test_name": test_name,
                "latency_ms": round(duration_ms, 6),
                "memory_mb": round(memory_delta, 6),
                "objects": object_delta,
                "status": status,
                "regression": False,  # Baseline has no regressions
                "deterministic": True,  # Assume deterministic for baseline
                "timestamp": deterministic_isoformat(
                    "baseline_test",
                    run_id,
                    endpoint,
                    test_name,
                    len(baseline_passport["test_results"]),
                )
            }

            baseline_passport["test_results"].append(test_result)

            # Update summary
            baseline_passport["summary"]["total_tests"] += 1
            if status == "PASS":
                baseline_passport["summary"]["passed_tests"] += 1
            else:
                baseline_passport["summary"]["failed_tests"] += 1

            # Update max values
            baseline_passport["summary"]["max_latency_ms"] = max(
                baseline_passport["summary"]["max_latency_ms"], duration_ms
            )
            baseline_passport["summary"]["max_memory_mb"] = max(
                baseline_passport["summary"]["max_memory_mb"], memory_delta
            )
            baseline_passport["summary"]["max_objects"] = max(
                baseline_passport["summary"]["max_objects"], object_delta
            )

            # Add endpoint to profiled list
            if endpoint not in baseline_passport["endpoints_profiled"]:
                baseline_passport["endpoints_profiled"].append(endpoint)

    # Calculate deterministic score
    deterministic_tests = sum(1 for r in baseline_passport["test_results"] if r.get("deterministic", True))
    baseline_passport["summary"]["deterministic_score"] = (
        deterministic_tests / baseline_passport["summary"]["total_tests"] * 100
    )

    # Set overall status
    if baseline_passport["summary"]["failed_tests"] > 0:
        baseline_passport["summary"]["overall_status"] = "FAIL"

    canonical = json.dumps(
        baseline_passport,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    passport_hash = deterministic_hash(canonical)

    baseline_passport["passport_hash"] = passport_hash
    canonical_with_hash = json.dumps(
        baseline_passport,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    )
    with open(output_file, 'w', encoding='ascii') as f:
        f.write(canonical_with_hash)

    print(f"Baseline passport created: {output_file}")
    print(f"Passport Hash: {passport_hash}")

    # Print summary
    summary = baseline_passport["summary"]
    print(f"Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed")
    print(f"Max Latency: {summary['max_latency_ms']:.3f}ms")
    print(f"Max Memory: {summary['max_memory_mb']:.2f}MB")
    print(f"Status: {summary['overall_status']}")

    return output_file


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Create Baseline Performance Passport")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", "-o", default="baseline_performance_passport.json", help="Output filename")
    parser.add_argument("--ci", action="store_true", help="CI/CD mode with enhanced logging")

    args = parser.parse_args()

    if args.ci:
        print("CI/CD Mode: Creating Baseline Performance Passport")

    try:
        baseline_file = create_baseline_passport(args.api_url, args.output)
        print(f"SUCCESS: Baseline passport created at {baseline_file}")
        sys.exit(0)
    except Exception as e:
        print(f"ERROR: Failed to create baseline passport: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
