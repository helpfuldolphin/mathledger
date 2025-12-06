#!/usr/bin/env python3
"""
CI Performance Check Script
Cursor B - Performance & Memory Sanity Cartographer

This script runs performance checks in CI environments, generates performance passports,
and fails PRs that exceed performance tolerances.

Global doctrine compliance:
- ASCII-only logs; no emojis in CI output
- Deterministic comparison via JSON hash
- Mechanical honesty: Status reflects API/test truth
"""

import json
import os
import sys
import time
import subprocess
import argparse
from datetime import datetime
from pathlib import Path


def run_command(cmd: str, cwd: str = None) -> tuple[int, str, str]:
    """
    Run a command and return (returncode, stdout, stderr).

    Args:
        cmd: Command to run
        cwd: Working directory

    Returns:
        Tuple of (returncode, stdout, stderr)
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=300  # 5 minute timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out after 5 minutes"


def check_api_availability(api_url: str, max_retries: int = 30) -> bool:
    """
    Check if the API is available and responding.

    Args:
        api_url: API base URL
        max_retries: Maximum number of retry attempts

    Returns:
        True if API is available, False otherwise
    """
    import requests

    print(f"Checking API availability at {api_url}...")

    for attempt in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=10)
            if response.status_code == 200:
                print("API is available and responding")
                return True
        except Exception as e:
            print(f"Attempt {attempt + 1}/{max_retries}: API not ready - {e}")
            time.sleep(2)

    print("ERROR: API is not available after maximum retries")
    return False


def run_performance_tests(api_url: str, baseline_file: str = None) -> tuple[bool, str]:
    """
    Run performance tests and generate passport.

    Args:
        api_url: API base URL
        baseline_file: Optional baseline passport file for comparison

    Returns:
        Tuple of (success, passport_file_path)
    """
    print("Running performance tests...")

    # Generate performance passport
    passport_cmd = f"python scripts/generate-performance-passport.py --api-url {api_url} --ci"
    if baseline_file:
        passport_cmd += f" --baseline {baseline_file}"

    returncode, stdout, stderr = run_command(passport_cmd)

    if returncode != 0:
        print(f"ERROR: Performance test failed with return code {returncode}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False, ""

    print("Performance tests completed successfully")
    print(stdout)

    # Find the generated passport file
    passport_file = "performance_passport.json"
    if not os.path.exists(passport_file):
        print("ERROR: Performance passport file not found")
        return False, ""

    return True, passport_file


def upload_artifacts(passport_file: str, artifacts_dir: str = "artifacts/performance") -> bool:
    """
    Upload performance passport as CI artifact.

    Args:
        passport_file: Path to the performance passport file
        artifacts_dir: Directory to store artifacts

    Returns:
        True if upload successful, False otherwise
    """
    print(f"Uploading artifacts to {artifacts_dir}...")

    try:
        # Create artifacts directory
        os.makedirs(artifacts_dir, exist_ok=True)

        # Copy passport to artifacts directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact_file = os.path.join(artifacts_dir, f"performance_passport_{timestamp}.json")

        import shutil
        shutil.copy2(passport_file, artifact_file)

        # Create latest symlink
        latest_file = os.path.join(artifacts_dir, "latest_performance_passport.json")
        if os.path.exists(latest_file):
            os.remove(latest_file)
        os.symlink(os.path.basename(artifact_file), latest_file)

        print(f"Artifacts uploaded successfully: {artifact_file}")
        return True

    except Exception as e:
        print(f"ERROR: Failed to upload artifacts: {e}")
        return False


def analyze_performance_passport(passport_file: str) -> dict:
    """
    Analyze the performance passport and return summary.

    Args:
        passport_file: Path to the performance passport file

    Returns:
        Dictionary with analysis results
    """
    try:
        with open(passport_file, 'r') as f:
            passport = json.load(f)

        summary = passport.get("summary", {})
        thresholds = passport.get("thresholds", {})

        analysis = {
            "overall_status": summary.get("overall_status", "UNKNOWN"),
            "total_tests": summary.get("total_tests", 0),
            "passed_tests": summary.get("passed_tests", 0),
            "failed_tests": summary.get("failed_tests", 0),
            "performance_regressions": summary.get("performance_regressions", 0),
            "memory_regressions": summary.get("memory_regressions", 0),
            "max_latency_ms": summary.get("max_latency_ms", 0.0),
            "max_memory_mb": summary.get("max_memory_mb", 0.0),
            "max_objects": summary.get("max_objects", 0),
            "deterministic_score": summary.get("deterministic_score", 0.0),
            "thresholds_exceeded": []
        }

        # Check threshold violations
        if analysis["max_latency_ms"] > thresholds.get("max_latency_threshold_ms", 10.0):
            analysis["thresholds_exceeded"].append(f"Latency: {analysis['max_latency_ms']:.3f}ms > {thresholds.get('max_latency_threshold_ms', 10.0)}ms")

        if analysis["max_memory_mb"] > thresholds.get("max_memory_threshold_mb", 10.0):
            analysis["thresholds_exceeded"].append(f"Memory: {analysis['max_memory_mb']:.2f}MB > {thresholds.get('max_memory_threshold_mb', 10.0)}MB")

        if analysis["max_objects"] > thresholds.get("max_objects_threshold", 1000):
            analysis["thresholds_exceeded"].append(f"Objects: {analysis['max_objects']} > {thresholds.get('max_objects_threshold', 1000)}")

        return analysis

    except Exception as e:
        print(f"ERROR: Failed to analyze performance passport: {e}")
        return {"overall_status": "ERROR", "error": str(e)}


def main():
    """Main function for CI performance checking."""
    parser = argparse.ArgumentParser(description="CI Performance Check")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--baseline", help="Baseline passport file for comparison")
    parser.add_argument("--artifacts-dir", default="artifacts/performance", help="Artifacts directory")
    parser.add_argument("--wait-timeout", type=int, default=60, help="API wait timeout in seconds")
    parser.add_argument("--no-upload", action="store_true", help="Skip artifact upload")

    args = parser.parse_args()

    print("CI Performance Check - Cursor B")
    print("=" * 50)
    print(f"API URL: {args.api_url}")
    print(f"Baseline: {args.baseline or 'None'}")
    print(f"Artifacts Dir: {args.artifacts_dir}")
    print(f"Wait Timeout: {args.wait_timeout}s")
    print("=" * 50)

    # Check API availability
    if not check_api_availability(args.api_url, args.wait_timeout // 2):
        print("FAIL: API is not available")
        sys.exit(1)

    # Run performance tests
    success, passport_file = run_performance_tests(args.api_url, args.baseline)
    if not success:
        print("FAIL: Performance tests failed")
        sys.exit(1)

    # Analyze performance passport
    analysis = analyze_performance_passport(passport_file)

    print("\nPerformance Analysis:")
    print("-" * 30)
    print(f"Overall Status: {analysis['overall_status']}")
    print(f"Tests: {analysis['passed_tests']}/{analysis['total_tests']} passed")
    print(f"Performance Regressions: {analysis['performance_regressions']}")
    print(f"Memory Regressions: {analysis['memory_regressions']}")
    print(f"Max Latency: {analysis['max_latency_ms']:.3f}ms")
    print(f"Max Memory: {analysis['max_memory_mb']:.2f}MB")
    print(f"Max Objects: {analysis['max_objects']}")
    print(f"Deterministic Score: {analysis['deterministic_score']:.1f}%")

    if analysis.get("thresholds_exceeded"):
        print("\nThreshold Violations:")
        for violation in analysis["thresholds_exceeded"]:
            print(f"  - {violation}")

    # Upload artifacts
    if not args.no_upload:
        upload_success = upload_artifacts(passport_file, args.artifacts_dir)
        if not upload_success:
            print("WARNING: Failed to upload artifacts")

    # Determine exit code
    if analysis["overall_status"] == "FAIL":
        print("\nFAIL: Performance regressions detected")
        sys.exit(1)
    elif analysis["overall_status"] == "ERROR":
        print("\nERROR: Failed to analyze performance")
        sys.exit(1)
    else:
        print("\nPASS: No performance regressions detected")
        sys.exit(0)


if __name__ == "__main__":
    main()
