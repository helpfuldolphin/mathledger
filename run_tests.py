#!/usr/bin/env python3
"""
Test runner for MathLedger v0.5 integration tests.
This script sets up the test environment and runs the comprehensive test suite.
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_test_environment():
    """Set up the test environment."""
    from backend.security.runtime_env import MissingEnvironmentVariable, get_database_url

    try:
        db_url = get_database_url()
    except MissingEnvironmentVariable as exc:
        print(f"❌ {exc}")
        sys.exit(2)

    os.environ["DATABASE_URL"] = db_url
    print(f"Using DATABASE_URL: {db_url}")

def run_tests():
    """Run the integration test suite."""
    setup_test_environment()

    # Run pytest with the integration test file
    cmd = [
        sys.executable, "-m", "pytest",
        "test_integration_v05.py",
        "-v",
        "--tb=short",
        "--color=yes"
    ]

    print("Running integration tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
