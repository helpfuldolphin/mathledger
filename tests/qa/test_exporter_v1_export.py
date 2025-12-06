#!/usr/bin/env python3
"""Smoke tests for export_fol_ab.py --export functionality.

Tests database export with actual PostgreSQL connection.
Requires DATABASE_URL environment variable to be set.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
import pytest

# Skip all tests if DATABASE_URL not set
pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="DATABASE_URL not set (integration test)"
)


def create_temp_jsonl(content_lines: list[str], encoding: str = 'utf-8') -> Path:
    """Create a temporary JSONL file with given content."""
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding=encoding)
    for line in content_lines:
        temp_file.write(line + '\n')
    temp_file.close()
    return Path(temp_file.name)


def run_exporter(input_file: Path, export: bool = False, dry_run: bool = False) -> tuple[int, str, str]:
    """Run the exporter and return (exit_code, stdout, stderr)."""
    cmd = ['python', 'backend/tools/export_fol_ab.py', '--input', str(input_file)]
    if export:
        cmd.append('--export')
    if dry_run:
        cmd.append('--dry-run')

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
    return result.returncode, result.stdout.strip(), result.stderr.strip()


def get_statement_count_by_content(content_norm: str) -> int:
    """Query database for statement count with SHA-256 hash of content_norm."""
    import hashlib
    import psycopg

    db_url = os.getenv("DATABASE_URL")
    # Hash the content_norm (not the V1 hash field) - matches export logic
    content_hash = hashlib.sha256(content_norm.encode('utf-8')).digest()

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM statements WHERE hash = %s", (content_hash,))
            return cur.fetchone()[0]


class TestExporterExport:
    """Test cases for --export database integration."""

    def test_export_smoke(self):
        """Smoke test: Export valid V1 records and verify insertion."""
        # Create unique content for this test run
        import time
        timestamp = str(time.time())

        content_1 = f"p -> p (test_smoke_{timestamp}_1)"
        content_2 = f"(p /\\ q) -> p (test_smoke_{timestamp}_2)"

        content = [
            json.dumps({
                "id": 1,
                "theory_id": 1,
                "hash": "ignored_hash_1",  # Hash field is ignored, content_norm is hashed
                "content_norm": content_1,
                "is_axiom": False
            }),
            json.dumps({
                "id": 2,
                "theory_id": 1,
                "hash": "ignored_hash_2",
                "content_norm": content_2,
                "is_axiom": False
            }),
        ]

        temp_file = create_temp_jsonl(content)
        try:
            # Run export
            exit_code, stdout, stderr = run_exporter(temp_file, export=True)

            # Should succeed
            assert exit_code == 0, f"Export should succeed, got: {stdout}"
            assert stdout.startswith("EXPORT ok:"), f"Expected 'EXPORT ok:' prefix, got: {stdout}"
            assert "2 records inserted" in stdout, f"Expected 2 records inserted, got: {stdout}"

            # Verify records were inserted (query by content_norm hash)
            count1 = get_statement_count_by_content(content_1)
            count2 = get_statement_count_by_content(content_2)

            assert count1 == 1, f"Expected 1 record for content 1, got {count1}"
            assert count2 == 1, f"Expected 1 record for content 2, got {count2}"

        finally:
            temp_file.unlink()

    def test_export_idempotent(self):
        """Test that running export twice doesn't duplicate records."""
        import time
        timestamp = str(time.time())

        content_norm = f"test idempotent ({timestamp})"

        content = [
            json.dumps({
                "id": 1,
                "theory_id": 1,
                "hash": "ignored_hash",
                "content_norm": content_norm,
                "is_axiom": False
            }),
        ]

        temp_file = create_temp_jsonl(content)
        try:
            # Run export twice
            exit_code1, stdout1, _ = run_exporter(temp_file, export=True)
            exit_code2, stdout2, _ = run_exporter(temp_file, export=True)

            # Both should succeed
            assert exit_code1 == 0, f"First export failed: {stdout1}"
            assert exit_code2 == 0, f"Second export failed: {stdout2}"

            # First run should insert 1
            assert "1 records inserted" in stdout1, f"Expected 1 record inserted on first run, got: {stdout1}"

            # Second run should report 0 inserted (1 skipped duplicate)
            assert "0 records inserted" in stdout2 and "skipped 1 duplicates" in stdout2, \
                f"Expected 0 inserted, 1 skipped on second run, got: {stdout2}"

            # Verify only 1 record exists in database
            count = get_statement_count_by_content(content_norm)
            assert count == 1, f"Expected exactly 1 record (no duplicates), got {count}"

        finally:
            temp_file.unlink()

    def test_export_with_dry_run_error(self):
        """Test that --export and --dry-run cannot be used together."""
        content = [
            json.dumps({
                "id": 1,
                "theory_id": 1,
                "hash": "test",
                "content_norm": "test",
                "is_axiom": False
            }),
        ]

        temp_file = create_temp_jsonl(content)
        try:
            exit_code, stdout, stderr = run_exporter(temp_file, export=True, dry_run=True)

            # Should fail
            assert exit_code != 0, "Should reject --export and --dry-run together"
            assert "error:" in stdout, f"Expected error message, got: {stdout}"
            assert "Cannot use" in stdout or "together" in stdout, \
                f"Expected mutual exclusivity message, got: {stdout}"

        finally:
            temp_file.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
