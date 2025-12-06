#!/usr/bin/env python3
"""
Dry-run SPARK test harness - No Docker required.

This script performs a lightweight check of the First Organism test suite
without attempting database connections or running tests.

It:
1. Imports the FO test module
2. Lists tests marked with @pytest.mark.first_organism
3. Checks if fixtures can be collected (without executing them)

Usage:
    python scripts/dry_run_spark.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
except ImportError:
    print("ERROR: pytest not installed. Install with: uv sync")
    sys.exit(1)


def collect_first_organism_tests(output_json: bool = False):
    """Collect all tests marked with @pytest.mark.first_organism."""
    result = {
        "success": False,
        "import_status": None,
        "test_collection": {"tests": [], "module_has_marker": False, "total_functions": 0},
        "fixtures": {"found": [], "missing": [], "environment_mode_available": False, "detect_function_available": False},
        "errors": []
    }
    
    if not output_json:
        print("=" * 70)
        print("SPARK Dry-Run Test Harness")
        print("=" * 70)
        print()

    # Import the test module
    if not output_json:
        print("[1/3] Importing test module...")
    try:
        from tests.integration import test_first_organism
        result["import_status"] = "success"
        if not output_json:
            print("  ✓ Successfully imported tests.integration.test_first_organism")
    except ImportError as e:
        result["import_status"] = "import_error"
        result["errors"].append(f"Failed to import test module: {e}")
        if not output_json:
            print(f"  ✗ Failed to import test module: {e}")
        if output_json:
            print(json.dumps(result, indent=2))
        return False
    except Exception as e:
        result["import_status"] = "error"
        result["errors"].append(f"Unexpected error importing test module: {e}")
        if not output_json:
            print(f"  ✗ Unexpected error importing test module: {e}")
            import traceback
            traceback.print_exc()
        if output_json:
            print(json.dumps(result, indent=2))
        return False

    if not output_json:
        print()

    # Collect tests using manual inspection (no DB connection)
    if not output_json:
        print("[2/3] Collecting tests marked with @pytest.mark.first_organism...")
    try:
        import inspect
        
        # Get all test functions from the module
        test_functions = [
            name
            for name, obj in inspect.getmembers(test_first_organism)
            if inspect.isfunction(obj) and name.startswith("test_")
        ]
        result["test_collection"]["total_functions"] = len(test_functions)
        
        # Check for first_organism marker on functions and module
        first_organism_tests = []
        module_has_marker = False
        
        # Check module-level marker
        if hasattr(test_first_organism, "pytestmark"):
            module_markers = test_first_organism.pytestmark
            if isinstance(module_markers, list):
                module_has_marker = any(
                    getattr(m, "name", None) == "first_organism"
                    or str(m).find("first_organism") >= 0
                    for m in module_markers
                )
        result["test_collection"]["module_has_marker"] = module_has_marker
        
        # Check each test function
        for name in test_functions:
            func = getattr(test_first_organism, name)
            has_marker = False
            
            # Check function-level markers
            if hasattr(func, "pytestmark"):
                markers = func.pytestmark
                if isinstance(markers, list):
                    has_marker = any(
                        getattr(m, "name", None) == "first_organism"
                        or str(m).find("first_organism") >= 0
                        for m in markers
                    )
            
            # If module has marker or function has marker, include it
            if module_has_marker or has_marker:
                first_organism_tests.append(name)
        
        result["test_collection"]["tests"] = sorted(first_organism_tests)
        
        if not output_json:
            if first_organism_tests:
                print(f"  ✓ Found {len(first_organism_tests)} test(s) marked first_organism:")
                for test_name in sorted(first_organism_tests):
                    print(f"    - {test_name}")
            else:
                print(f"  ⚠ No tests explicitly marked (found {len(test_functions)} test function(s) total)")
                if module_has_marker:
                    print("    → Module has first_organism marker, all tests inherit it")
                    print(f"    → Test functions: {', '.join(sorted(test_functions[:5]))}")
                    if len(test_functions) > 5:
                        print(f"    → ... and {len(test_functions) - 5} more")

    except Exception as e:
        result["errors"].append(f"Error collecting tests: {e}")
        if not output_json:
            print(f"  ✗ Error collecting tests: {e}")
            import traceback
            traceback.print_exc()
        if output_json:
            print(json.dumps(result, indent=2))
        return False

    if not output_json:
        print()

    # Check fixture collection (without executing)
    if not output_json:
        print("[3/3] Checking fixture collection (dry-run, no DB connection)...")
    try:
        from tests.integration import conftest
        
        # List available fixtures
        import inspect
        fixtures = [
            name
            for name, obj in inspect.getmembers(conftest)
            if inspect.isfunction(obj) and hasattr(obj, "_pytestfixturefunction")
        ]
        
        # Check for key FO fixtures
        key_fixtures = [
            "first_organism_db",
            "first_organism_env",
            "first_organism_attestation_context",
            "environment_mode",
            "test_db_url",
        ]
        
        found_fixtures = []
        missing_fixtures = []
        for fixture_name in key_fixtures:
            if fixture_name in fixtures:
                found_fixtures.append(fixture_name)
            else:
                # Check if it's defined but not detected
                if hasattr(conftest, fixture_name):
                    found_fixtures.append(fixture_name)
                else:
                    missing_fixtures.append(fixture_name)
        
        result["fixtures"]["found"] = sorted(found_fixtures)
        result["fixtures"]["missing"] = sorted(missing_fixtures)
        
        if not output_json:
            if found_fixtures:
                print(f"  ✓ Found {len(found_fixtures)} key fixture(s):")
                for fix in sorted(found_fixtures):
                    print(f"    - {fix}")
            
            if missing_fixtures:
                print(f"  ⚠ Missing {len(missing_fixtures)} fixture(s):")
                for fix in sorted(missing_fixtures):
                    print(f"    - {fix}")
        
        # Check EnvironmentMode class
        if hasattr(conftest, "EnvironmentMode"):
            env_mode = conftest.EnvironmentMode
            result["fixtures"]["environment_mode_available"] = True
            if not output_json:
                print(f"  ✓ EnvironmentMode class available")
                if hasattr(env_mode, "chain_status"):
                    print(f"    → Chain status values: READY, ABSTAIN, SKIP, MOCK")
        
        # Check detect_environment_mode function
        if hasattr(conftest, "detect_environment_mode"):
            result["fixtures"]["detect_function_available"] = True
            if not output_json:
                print(f"  ✓ detect_environment_mode function available")
                print(f"    → (Not calling to avoid DB connection)")
        
    except ImportError as e:
        result["errors"].append(f"Failed to import conftest: {e}")
        if not output_json:
            print(f"  ✗ Failed to import conftest: {e}")
        if output_json:
            print(json.dumps(result, indent=2))
        return False
    except Exception as e:
        result["errors"].append(f"Error checking fixtures: {e}")
        if not output_json:
            print(f"  ✗ Error checking fixtures: {e}")
            import traceback
            traceback.print_exc()
        if output_json:
            print(json.dumps(result, indent=2))
        return False

    result["success"] = True
    
    if output_json:
        print(json.dumps(result, indent=2))
        return True
    
    print()
    print("=" * 70)
    print("Dry-run complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Set FIRST_ORGANISM_TESTS=true to enable tests")
    print("  2. Ensure DATABASE_URL is set (or tests will skip gracefully)")
    print("  3. Run: FIRST_ORGANISM_TESTS=true pytest tests/integration/test_first_organism.py -v")
    print()
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dry-run SPARK test harness - No Docker required"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable text"
    )
    args = parser.parse_args()
    
    success = collect_first_organism_tests(output_json=args.json)
    sys.exit(0 if success else 1)

