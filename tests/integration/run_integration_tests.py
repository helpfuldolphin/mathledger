#!/usr/bin/env python3
"""
Run integration tests for MathLedger API.
This script sets up the test environment and runs the integration test suite.
"""

import subprocess
import sys
import os
import time


def check_docker():
    """Check if Docker is available."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def run_integration_tests():
    """Run the integration test suite."""
    if not check_docker():
        print("Error: Docker is required for integration tests")
        print("Please install Docker and ensure it's running")
        return False

    print("Starting MathLedger API integration tests...")
    print("This will start a PostgreSQL container and run comprehensive tests")

    # Change to project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(project_root)

    try:
        # Run pytest with integration tests
        result = subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/integration/",
            "-v",
            "--tb=short",
            "--color=yes"
        ], check=True)

        print("\n✅ Integration tests completed successfully!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Integration tests failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⚠️  Integration tests interrupted by user")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
