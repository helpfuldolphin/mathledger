#!/usr/bin/env python3
"""
test_golden_run.py
Validation script for Golden Run automation system.
Tests database connectivity, schema compatibility, and progress logging.
"""

import os
import sys
import psycopg
from pathlib import Path
from typing import Dict, Any, Tuple

def test_database_connectivity() -> Tuple[bool, str]:
    """Test database connection and basic query execution."""
    try:
        db_url = os.getenv("DATABASE_URL", "postgresql://ml:mlpass@localhost:5432/mathledger")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                if result and result[0] == 1:
                    return True, "Database connection successful"
                else:
                    return False, "Database query returned unexpected result"
    except Exception as e:
        return False, f"Database connection failed: {e}"

def test_schema_compatibility() -> Tuple[bool, str]:
    """Test if database schema is compatible with progress.py queries."""
    try:
        db_url = os.getenv("DATABASE_URL", "postgresql://ml:mlpass@localhost:5432/mathledger")
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                # Test blocks table schema
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'blocks'
                    AND column_name IN ('block_number', 'merkle_root', 'header', 'created_at')
                    ORDER BY column_name
                """)
                block_columns = {row[0]: row[1] for row in cur.fetchall()}

                required_block_columns = {'block_number', 'merkle_root', 'header', 'created_at'}
                missing_block_columns = required_block_columns - set(block_columns.keys())

                if missing_block_columns:
                    return False, f"Missing block columns: {missing_block_columns}"

                # Test statements table schema
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'statements'
                    AND column_name IN ('text', 'content_norm', 'system_id', 'is_axiom', 'derivation_depth')
                    ORDER BY column_name
                """)
                statement_columns = {row[0]: row[1] for row in cur.fetchall()}

                # At least one of text or content_norm should exist
                if not ('text' in statement_columns or 'content_norm' in statement_columns):
                    return False, "Missing required statement columns: need either 'text' or 'content_norm'"

                # Test proofs table schema
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = 'proofs'
                    AND column_name IN ('status', 'success', 'statement_id')
                    ORDER BY column_name
                """)
                proof_columns = {row[0]: row[1] for row in cur.fetchall()}

                # At least one of status or success should exist
                if not ('status' in proof_columns or 'success' in proof_columns):
                    return False, "Missing required proof columns: need either 'status' or 'success'"

                return True, f"Schema compatible - blocks: {list(block_columns.keys())}, statements: {list(statement_columns.keys())}, proofs: {list(proof_columns.keys())}"

    except Exception as e:
        return False, f"Schema compatibility test failed: {e}"

def test_progress_md_permissions() -> Tuple[bool, str]:
    """Test if docs/progress.md can be written to."""
    try:
        progress_path = Path("docs/progress.md")

        # Create docs directory if it doesn't exist
        progress_path.parent.mkdir(exist_ok=True)

        # Test write permission
        test_content = "# Test entry\nThis is a test.\n"
        with open(progress_path, 'a', encoding='utf-8') as f:
            f.write(test_content)

        # Verify content was written
        with open(progress_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if test_content in content:
                # Remove test content
                clean_content = content.replace(test_content, "")
                with open(progress_path, 'w', encoding='utf-8') as f:
                    f.write(clean_content)
                return True, f"Progress.md write test successful at {progress_path.absolute()}"
            else:
                return False, "Progress.md write test failed - content not found"

    except Exception as e:
        return False, f"Progress.md permissions test failed: {e}"

def test_progress_py_import() -> Tuple[bool, str]:
    """Test if progress.py can be imported and basic functions work."""
    try:
        # Add backend to path
        sys.path.insert(0, str(Path(__file__).parent / "backend"))

        from tools.progress import get_latest_run_data, append_latest_to_progress, _is_block_already_logged

        # Test function imports
        if not callable(get_latest_run_data):
            return False, "get_latest_run_data function not callable"
        if not callable(append_latest_to_progress):
            return False, "append_latest_to_progress function not callable"
        if not callable(_is_block_already_logged):
            return False, "_is_block_already_logged function not callable"

        return True, "Progress.py import and function validation successful"

    except Exception as e:
        return False, f"Progress.py import test failed: {e}"

def test_idempotency_logic() -> Tuple[bool, str]:
    """Test idempotency logic without actually writing to progress.md."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "backend"))
        from tools.progress import _is_block_already_logged

        # Test with non-existent file
        result = _is_block_already_logged("non_existent_file.md", 123)
        if result != False:
            return False, "Idempotency test failed: should return False for non-existent file"

        # Test with None block_number
        result = _is_block_already_logged("docs/progress.md", None)
        if result != False:
            return False, "Idempotency test failed: should return False for None block_number"

        return True, "Idempotency logic validation successful"

    except Exception as e:
        return False, f"Idempotency test failed: {e}"

def main():
    """Run all validation tests and report results."""
    tests = [
        ("Database Connectivity", test_database_connectivity),
        ("Schema Compatibility", test_schema_compatibility),
        ("Progress.md Permissions", test_progress_md_permissions),
        ("Progress.py Import", test_progress_py_import),
        ("Idempotency Logic", test_idempotency_logic),
    ]

    print("=== GOLDEN RUN VALIDATION TESTS ===\n")

    all_passed = True
    results = {}

    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        try:
            passed, message = test_func()
            results[test_name] = (passed, message)
            status = "PASS" if passed else "FAIL"
            print(f"  {status}: {message}")
            if not passed:
                all_passed = False
        except Exception as e:
            results[test_name] = (False, f"Test crashed: {e}")
            print(f"  FAIL: Test crashed: {e}")
            all_passed = False
        print()

    print("=== SUMMARY ===")
    for test_name, (passed, message) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall Status: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    if not all_passed:
        print("\nRecommendations:")
        for test_name, (passed, message) in results.items():
            if not passed:
                print(f"- Fix {test_name}: {message}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
