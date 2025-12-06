#!/usr/bin/env python3
"""
verify_local_schema.py - Validate local PostgreSQL schema for Golden Run
"""

import os
import sys
import psycopg
from pathlib import Path

def test_db_connectivity():
    """Test database connectivity via DATABASE_URL"""
    try:
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            return False, "DATABASE_URL environment variable not set"

        conn = psycopg.connect(db_url)
        cur = conn.cursor()
        cur.execute("SELECT 1")
        result = cur.fetchone()
        conn.close()

        if result and result[0] == 1:
            return True, f"Connected to {db_url}"
        else:
            return False, "Query returned unexpected result"
    except Exception as e:
        return False, f"Connection failed: {e}"

def test_blocks_schema():
    """Test if blocks table has required columns"""
    try:
        db_url = os.getenv("DATABASE_URL")
        conn = psycopg.connect(db_url)
        cur = conn.cursor()

        # Check if blocks table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_name = 'blocks'
            )
        """)
        table_exists = cur.fetchone()[0]

        if not table_exists:
            conn.close()
            return False, "blocks table does not exist"

        # Check required columns
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'blocks'
            AND column_name IN ('block_number', 'merkle_root', 'created_at', 'header')
            ORDER BY column_name
        """)
        columns = {row[0]: row[1] for row in cur.fetchall()}

        required_columns = {'block_number', 'merkle_root', 'created_at', 'header'}
        missing_columns = required_columns - set(columns.keys())

        conn.close()

        if missing_columns:
            return False, f"Missing columns in blocks table: {missing_columns}"
        else:
            return True, f"blocks table has required columns: {list(columns.keys())}"

    except Exception as e:
        return False, f"Schema check failed: {e}"

def test_progress_md_writable():
    """Test if docs/progress.md can be written to"""
    try:
        progress_path = Path("docs/progress.md")

        # Create docs directory if it doesn't exist
        progress_path.parent.mkdir(exist_ok=True)

        # Test write permission
        test_content = "# Test entry for validation\n"
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
                return True, f"docs/progress.md is writable at {progress_path.absolute()}"
            else:
                return False, "Write test failed - content not found"

    except Exception as e:
        return False, f"Progress.md write test failed: {e}"

def test_idempotent_append():
    """Test idempotent append logic"""
    try:
        # Add backend to path
        sys.path.insert(0, str(Path(__file__).parent / "backend"))

        from tools.progress import _is_block_already_logged

        # Test with non-existent file
        result = _is_block_already_logged("non_existent_file.md", 123)
        if result != False:
            return False, "Should return False for non-existent file"

        # Test with None block_number
        result = _is_block_already_logged("docs/progress.md", None)
        if result != False:
            return False, "Should return False for None block_number"

        # Test with existing file (create test content)
        test_file = Path("test_progress.md")
        test_content = "## [2025-01-13 14:30] Block 123\n- merkle_root: abc123\n"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_content)

        # Test idempotency
        result = _is_block_already_logged(str(test_file), 123)
        if result != True:
            return False, "Should return True for existing block number"

        # Test with different block number
        result = _is_block_already_logged(str(test_file), 456)
        if result != False:
            return False, "Should return False for different block number"

        # Cleanup
        test_file.unlink()

        return True, "Idempotent append logic works correctly"

    except Exception as e:
        return False, f"Idempotent append test failed: {e}"

def main():
    """Run all validation tests"""
    tests = [
        ("DB Connectivity", test_db_connectivity),
        ("Blocks Schema", test_blocks_schema),
        ("Progress.md Writable", test_progress_md_writable),
        ("Idempotent Append", test_idempotent_append),
    ]

    print("=== LOCAL SCHEMA VALIDATION ===\n")

    all_passed = True
    results = {}

    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
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
