#!/usr/bin/env python3
"""
Simple test runner for MathLedger v0.5 that works without external dependencies.
This ensures the test suite can run in any environment.
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
        "test_v05_integration.py",
        "-v",
        "--tb=short",
        "--color=yes",
        "--maxfail=1"
    ]

    print("Running MathLedger v0.5 integration tests...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✅ All tests passed! MathLedger v0.5 is fully functional.")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code: {e.returncode}")
        return e.returncode

if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
