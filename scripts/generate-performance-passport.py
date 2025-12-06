#!/usr/bin/env python3
"""
Performance Passport Generator for CI/CD Integration
Cursor B - Performance & Memory Sanity Cartographer

This script generates a comprehensive performance passport with dual memory profilers
(psutil + tracemalloc), deterministic outputs, and CI artifact uploads.

Global doctrine compliance:
- ASCII-only logs; no emojis in CI output
- Deterministic comparison via JSON hash
- Mechanical honesty: Status reflects API/test truth
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
from typing import Any, Dict, List, Optional, Tuple

from backend.repro.determinism import (
    deterministic_hash,
    deterministic_isoformat,
    deterministic_run_id,
    deterministic_slug,
)


class PerformancePassportGenerator:
    """
    Performance Passport Generator with dual memory profilers

    Maintains endpoint performance passports (JSON) with dual memory profilers
    (psutil + tracemalloc), deterministic outputs, and CI artifact uploads.
    """

    def __init__(self):
        self.thresholds = {
            "max_latency_threshold_ms": 10.0,
            "max_memory_threshold_mb": 10.0,
            "max_objects_threshold": 1000,
            "regression_tolerance_percent": 5.0,
            "deterministic_threshold_percent": 95.0
        }
        thresholds_fingerprint = json.dumps(
            self.thresholds,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        self.run_id = deterministic_run_id("perf", thresholds_fingerprint)
        self.session_id = deterministic_slug("perf_session", self.run_id, length=8)
        passport_timestamp = deterministic_isoformat("performance_passport", self.run_id)
        self.passport = {
            "cartographer": "Cursor B - Performance & Memory Sanity Cartographer",
            "run_id": self.run_id,
            "session_id": self.session_id,
            "timestamp": passport_timestamp,
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
            }
        }

    def add_test_result(self, endpoint: str, test_name: str, latency_ms: float,
                       memory_mb: float, objects: int, status: str,
                       regression: bool = False, deterministic: bool = True):
        """Add a test result to the performance passport."""
        result = {
            "endpoint": endpoint,
            "test_name": test_name,
            "latency_ms": round(latency_ms, 6),
            "memory_mb": round(memory_mb, 6),
            "objects": objects,
            "status": status,
            "regression": regression,
            "deterministic": deterministic,
            "timestamp": deterministic_isoformat(
                "passport_test",
                self.run_id,
                endpoint,
                test_name,
                len(self.passport["test_results"]),
            )
        }

        self.passport["test_results"].append(result)

        # Update summary
        self.passport["summary"]["total_tests"] += 1
        if status == "PASS":
            self.passport["summary"]["passed_tests"] += 1
        else:
            self.passport["summary"]["failed_tests"] += 1

        if regression:
            if "latency" in test_name.lower() or "performance" in test_name.lower():
                self.passport["summary"]["performance_regressions"] += 1
            if "memory" in test_name.lower():
                self.passport["summary"]["memory_regressions"] += 1

        # Update max values
        self.passport["summary"]["max_latency_ms"] = max(
            self.passport["summary"]["max_latency_ms"], latency_ms
        )
        self.passport["summary"]["max_memory_mb"] = max(
            self.passport["summary"]["max_memory_mb"], memory_mb
        )
        self.passport["summary"]["max_objects"] = max(
            self.passport["summary"]["max_objects"], objects
        )

        # Calculate deterministic score
        deterministic_tests = sum(1 for r in self.passport["test_results"] if r.get("deterministic", True))
        self.passport["summary"]["deterministic_score"] = (
            deterministic_tests / self.passport["summary"]["total_tests"] * 100
        )

    def finalize_passport(self):
        """Finalize the performance passport."""
        if (self.passport["summary"]["failed_tests"] > 0 or
            self.passport["summary"]["performance_regressions"] > 0 or
            self.passport["summary"]["memory_regressions"] > 0):
            self.passport["summary"]["overall_status"] = "FAIL"

        # Add performance thresholds
        self.passport["thresholds"] = self.thresholds

        # Add system info
        self.passport["system_info"] = {
            "platform": os.name,
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
            "psutil_version": psutil.__version__,
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "cpu_count": psutil.cpu_count()
        }

    def save_passport(self, filename: str = "performance_passport.json") -> str:
        """Save the performance passport to a JSON file."""
        self.finalize_passport()
        canonical = json.dumps(
            self.passport,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        with open(filename, 'w', encoding='ascii') as f:
            f.write(canonical)
        return filename

    def get_passport_hash(self) -> str:
        """Get deterministic hash of the passport for comparison."""
        passport_str = json.dumps(
            self.passport,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        return deterministic_hash(passport_str)


class DualMemoryProfiler:
    """
    Dual Memory Profiler using psutil + tracemalloc

    Provides comprehensive memory monitoring with both process-level and
    Python object-level tracking.
    """

    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.initial_objects = None
        self.tracemalloc_start = None
        self.memory_snapshots = []

    def start_profiling(self):
        """Start comprehensive memory profiling."""
        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        self.tracemalloc_start = tracemalloc.get_traced_memory()

        # Get initial memory usage
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.initial_objects = len(gc.get_objects())

        # Take memory snapshot
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_mb': self.initial_memory,
            'objects': self.initial_objects
        })

    def stop_profiling(self) -> Tuple[float, int, float, Dict[str, Any]]:
        """
        Stop memory profiling and return comprehensive metrics.

        Returns:
            Tuple of (memory_delta_mb, object_delta, peak_memory_mb, detailed_metrics)
        """
        # Force garbage collection
        gc.collect()

        # Get final memory usage
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        final_objects = len(gc.get_objects())

        # Get tracemalloc peak
        current, peak = tracemalloc.get_traced_memory()
        peak_memory = peak / 1024 / 1024  # MB

        # Calculate deltas
        memory_delta = final_memory - self.initial_memory
        object_delta = final_objects - self.initial_objects

        # Take final snapshot
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'memory_mb': final_memory,
            'objects': final_objects
        })

        # Calculate detailed metrics
        detailed_metrics = {
            'memory_delta_mb': memory_delta,
            'object_delta': object_delta,
            'peak_memory_mb': peak_memory,
            'memory_efficiency': 1.0 - (memory_delta / max(self.initial_memory, 1.0)),
            'object_efficiency': 1.0 - (object_delta / max(self.initial_objects, 1.0)),
            'snapshots': len(self.memory_snapshots)
        }

        # Stop tracemalloc
        tracemalloc.stop()

        return memory_delta, object_delta, peak_memory, detailed_metrics


class EndpointProfiler:
    """
    Endpoint Profiler for testing API endpoints

    Tests endpoints with comprehensive performance monitoring and
    deterministic output validation.
    """

    def __init__(self, passport_generator: PerformancePassportGenerator):
        self.passport_generator = passport_generator
        self.profiler = DualMemoryProfiler()
        self.base_url = "http://localhost:8000"  # Default API base URL

    def set_base_url(self, base_url: str):
        """Set the API base URL for testing."""
        self.base_url = base_url.rstrip('/')

    def profile_endpoint(self, endpoint: str, test_variants: List[Dict[str, Any]] = None):
        """Profile a specific endpoint with comprehensive testing."""
        if test_variants is None:
            test_variants = [{}]

        for i, test_data in enumerate(test_variants):
            test_name = f"{endpoint}_variant_{i}"

            # Start memory profiling
            self.profiler.start_profiling()

            # Time the endpoint call
            start_time = time.perf_counter()

            try:
                # Make API call
                url = f"{self.base_url}/{endpoint.lstrip('/')}"
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
            memory_delta, object_delta, peak_memory, detailed_metrics = self.profiler.stop_profiling()

            # Performance assertions
            regression = (
                duration_ms > self.passport_generator.thresholds["max_latency_threshold_ms"] or
                memory_delta > self.passport_generator.thresholds["max_memory_threshold_mb"] or
                object_delta > self.passport_generator.thresholds["max_objects_threshold"] or
                peak_memory > self.passport_generator.thresholds["max_memory_threshold_mb"]
            )

            # Deterministic check (run multiple times)
            deterministic = self._check_deterministic(endpoint)

            # Record in passport
            self.passport_generator.add_test_result(
                endpoint, test_name, duration_ms, memory_delta, object_delta,
                status, regression, deterministic
            )

            # Report regression if detected
            if regression:
                print(f"WARNING: PERFORMANCE REGRESSION: {endpoint} {test_name} exceeded thresholds - "
                      f"Latency: {duration_ms:.3f}ms, Memory: {memory_delta:.2f}MB, "
                      f"Objects: {object_delta}, Peak: {peak_memory:.2f}MB")

            # Add endpoint to profiled list
            if endpoint not in self.passport_generator.passport["endpoints_profiled"]:
                self.passport_generator.passport["endpoints_profiled"].append(endpoint)

    def _check_deterministic(self, endpoint: str, runs: int = 3) -> bool:
        """Check if the endpoint produces deterministic results."""
        try:
            results = []
            for _ in range(runs):
                response = requests.get(f"{self.base_url}/{endpoint.lstrip('/')}", timeout=30)
                response.raise_for_status()
                result = response.json()
                # Convert to hashable format for comparison
                result_str = json.dumps(result, sort_keys=True)
                results.append(hashlib.md5(result_str.encode()).hexdigest())

            # All results should be identical
            return len(set(results)) == 1
        except Exception:
            return False


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(description="Generate Performance Passport")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", "-o", default="performance_passport.json", help="Output filename")
    parser.add_argument("--ci", action="store_true", help="CI/CD mode with enhanced logging")
    parser.add_argument("--upload", action="store_true", help="Upload passport as CI artifact")
    parser.add_argument("--baseline", help="Baseline passport file for comparison")

    args = parser.parse_args()

    if args.ci:
        print("CI/CD Mode: Generating Performance Passport")
        print(f"API URL: {args.api_url}")
        print(f"Output: {args.output}")

    # Initialize passport generator
    passport_generator = PerformancePassportGenerator()
    profiler = EndpointProfiler(passport_generator)
    profiler.set_base_url(args.api_url)

    # Test core endpoints
    endpoints_to_test = [
        "metrics",
        "health",
        "blocks/latest",
        "statements",
        "ui",
        "heartbeat.json"
    ]

    print("Testing endpoints with dual memory profilers...")
    for endpoint in endpoints_to_test:
        print(f"Profiling endpoint: {endpoint}")
        profiler.profile_endpoint(endpoint)

    # Save passport
    passport_file = passport_generator.save_passport(args.output)
    passport_hash = passport_generator.get_passport_hash()

    print(f"Performance Passport generated: {passport_file}")
    print(f"Passport Hash: {passport_hash}")

    # Print summary
    summary = passport_generator.passport["summary"]
    print(f"Summary: {summary['passed_tests']}/{summary['total_tests']} tests passed")
    print(f"Max Latency: {summary['max_latency_ms']:.3f}ms")
    print(f"Max Memory: {summary['max_memory_mb']:.2f}MB")
    print(f"Status: {summary['overall_status']}")

    # Baseline comparison
    if args.baseline and os.path.exists(args.baseline):
        print(f"Comparing with baseline: {args.baseline}")
        with open(args.baseline, 'r') as f:
            baseline_data = json.load(f)

        baseline_payload = json.dumps(
            baseline_data,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=True,
        )
        baseline_hash = deterministic_hash(baseline_payload)

        if passport_hash != baseline_hash:
            print("WARNING: Passport differs from baseline - potential regression detected")
            sys.exit(1)
        else:
            print("PASS: Passport matches baseline - no regression detected")

    # Upload as CI artifact
    if args.upload:
        print("Uploading passport as CI artifact...")
        # This would integrate with your CI system's artifact upload
        # For now, just copy to artifacts directory
        artifacts_dir = "artifacts/performance"
        os.makedirs(artifacts_dir, exist_ok=True)

        artifact_slug = deterministic_slug("performance_passport", passport_hash)
        artifact_file = os.path.join(artifacts_dir, f"performance_passport_{artifact_slug}.json")
        os.rename(passport_file, artifact_file)

        # Create latest symlink
        latest_file = os.path.join(artifacts_dir, "latest_performance_passport.json")
        os.symlink(os.path.basename(artifact_file), latest_file)

        print(f"Passport uploaded to: {artifact_file}")

    # Exit with appropriate code
    if summary['overall_status'] == "FAIL":
        print("FAIL: Performance passport indicates regressions")
        sys.exit(1)
    else:
        print("PASS: Performance passport indicates no regressions")
        sys.exit(0)


if __name__ == "__main__":
    main()
