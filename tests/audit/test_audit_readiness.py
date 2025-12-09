import json
import os
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path
import sqlite3

# Add project root to path to allow importing from 'scripts'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_audit_readiness import generate_completeness_report, REQUIRED_DATA_SOURCES

SCRIPT_DIR = PROJECT_ROOT / 'scripts'

class TestAuditReadiness(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and an in-memory SQLite database."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_dir = Path(self.temp_dir.name)
        
        # Using SQLite's in-memory DB for hermetic testing
        # We need to monkey-patch the check functions to work with sqlite3
        self.db_conn = sqlite3.connect(":memory:")
        self.patch_sqlite()

        # Paths for reports
        self.ready_report_path = self.log_dir / 'ready_report.json'
        self.incomplete_report_path = self.log_dir / 'incomplete_report.json'
        self.auditor_output_path = self.log_dir / 'auditor_output.json'

    def tearDown(self):
        """Clean up the temporary directory and close DB connection."""
        self.db_conn.close()
        self.temp_dir.cleanup()

    def patch_sqlite(self):
        """
        Monkey-patch the DB check functions to work with sqlite3 syntax.
        This is a common testing pattern to adapt logic for different DB backends.
        """
        from scripts import check_audit_readiness
        
        def mock_check_db_tables(conn: sqlite3.Connection):
            status_report = []
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            existing_tables = {row[0] for row in cursor.fetchall()}
            
            for table_name in REQUIRED_DATA_SOURCES["tables"]:
                status = "FOUND" if table_name in existing_tables else "NOT_FOUND"
                status_report.append({"source_name": table_name, "status": status})
            return status_report

        def mock_check_prerequisites(conn: sqlite3.Connection):
            # SQLite doesn't have extensions in the same way, so we just return MET
            return [{"check_name": "DB_EXTENSION_PG_TRGM", "status": "MET"}]

        # Replace the functions in the imported module with our mocks
        check_audit_readiness.check_db_tables = mock_check_db_tables
        check_audit_readiness.check_prerequisites = mock_check_prerequisites

    def create_all_tables(self):
        """Helper to create all required tables in the in-memory DB."""
        cursor = self.db_conn.cursor()
        for table_name in REQUIRED_DATA_SOURCES["tables"]:
            cursor.execute(f"CREATE TABLE {table_name} (id INTEGER PRIMARY KEY);")
        self.db_conn.commit()

    def create_required_files(self):
        """Helper to create all required files in the temp directory."""
        for file_name in REQUIRED_DATA_SOURCES["files"]:
            (self.log_dir / file_name).touch()

    def run_auditor_script(self, completeness_report_path: Path):
        """Helper to run the main auditor script."""
        script_path = SCRIPT_DIR / 'proof_dag_u2_audit.py'
        self.assertTrue(script_path.exists())
        return subprocess.run(
            [
                sys.executable, str(script_path),
                '--exp-id', 'test_exp',
                '--completeness-report', str(completeness_report_path)
            ],
            capture_output=True, text=True, check=False
        )

    def test_readiness_success_path(self):
        """1. Test readiness check for a fully READY system."""
        # Arrange: Create all DB tables and all required files
        self.create_all_tables()
        self.create_required_files()

        # Act: Generate the completeness report
        report = generate_completeness_report("test_exp", self.db_conn, self.log_dir)

        # Assert: The status is READY and all sources are FOUND
        self.assertEqual(report['completeness_status'], 'READY')
        self.assertTrue(all(s['status'] == 'FOUND' for s in report['data_source_status']))

    def test_readiness_fail_missing_table(self):
        """2. Test readiness check fails correctly when a table is missing."""
        # Arrange: Create files but not tables
        self.create_required_files()

        # Act: Generate the report
        report = generate_completeness_report("test_exp", self.db_conn, self.log_dir)

        # Assert: The status is INCOMPLETE and the missing tables are flagged
        self.assertEqual(report['completeness_status'], 'INCOMPLETE')
        missing_table_reports = [s for s in report['data_source_status'] if s['status'] == 'NOT_FOUND']
        self.assertGreater(len(missing_table_reports), 0)
        self.assertEqual(missing_table_reports[0]['source_name'], REQUIRED_DATA_SOURCES['tables'][0])

    def test_readiness_fail_missing_file(self):
        """3. Test readiness check fails correctly when a file is missing."""
        # Arrange: Create tables but not files
        self.create_all_tables()

        # Act: Generate the report
        report = generate_completeness_report("test_exp", self.db_conn, self.log_dir)
        
        # Assert: The status is INCOMPLETE and the missing files are flagged
        self.assertEqual(report['completeness_status'], 'INCOMPLETE')
        missing_file_reports = [s for s in report['data_source_status'] if s['status'] == 'NOT_FOUND']
        self.assertGreater(len(missing_file_reports), 0)
        self.assertEqual(missing_file_reports[0]['source_name'], REQUIRED_DATA_SOURCES['files'][0])

    def test_auditor_integration_with_reports(self):
        """4. Test that the main auditor respects the generated readiness reports."""
        # --- Part 1: Auditor fails with an INCOMPLETE report ---
        # Arrange: Create an incomplete state (missing tables) and generate the report
        self.create_required_files()
        incomplete_report_data = generate_completeness_report("test_exp", self.db_conn, self.log_dir)
        with open(self.incomplete_report_path, 'w') as f:
            json.dump(incomplete_report_data, f)

        # Act: Run the auditor pointing to the incomplete report
        fail_process = self.run_auditor_script(self.incomplete_report_path)

        # Assert: The auditor exits with a non-zero code and prints an error
        self.assertNotEqual(fail_process.returncode, 0)
        self.assertIn("Audit system is not ready", fail_process.stderr)
        self.assertIn("INCOMPLETE", fail_process.stderr)
        
        # --- Part 2: Auditor succeeds with a READY report ---
        # Arrange: Create a complete state and generate the report
        self.create_all_tables()
        ready_report_data = generate_completeness_report("test_exp", self.db_conn, self.log_dir)
        with open(self.ready_report_path, 'w') as f:
            json.dump(ready_report_data, f)
            
        # Act: Run the auditor pointing to the ready report
        success_process = self.run_auditor_script(self.ready_report_path)

        # Assert: The auditor runs successfully
        self.assertEqual(success_process.returncode, 0, f"Auditor failed unexpectedly: {success_process.stderr}")

if __name__ == '__main__':
    unittest.main()