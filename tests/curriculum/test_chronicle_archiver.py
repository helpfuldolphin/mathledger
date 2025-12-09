# PHASE IV — NOT USED IN PHASE I
# File: tests/curriculum/test_chronicle_archiver.py
import unittest
import os
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

def create_mock_report_file(dir_path, timestamp, critical=0, warning=0, info=0):
    """Helper to create a mock drift report file."""
    report = {
        "report_generated_utc": timestamp.isoformat(),
        "summary": {"severity_counts": {"CRITICAL": critical, "WARNING": warning, "INFO": info}},
        "drifts": [{'severity': 'CRITICAL'}] * critical + [{'severity': 'WARNING'}] * warning
    }
    file_path = os.path.join(dir_path, f"report_{timestamp.isoformat().replace(':', '-')}.json")
    with open(file_path, 'w') as f:
        json.dump(report, f)
    return file_path

class TestChronicleArchiver(unittest.TestCase):
    def setUp(self):
        self.test_dir = "tests/curriculum/temp_archiver_test"
        self.reports_dir = os.path.join(self.test_dir, "reports")
        self.artifacts_dir = os.path.join(self.test_dir, "artifacts")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.script_path = os.path.abspath("scripts/curriculum_drift_chronicle.py")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_script_ok_status(self):
        """Test the script with reports that should result in an OK status."""
        now = datetime.now()
        create_mock_report_file(self.reports_dir, now - timedelta(days=1), warning=1)
        create_mock_report_file(self.reports_dir, now, info=2)
        
        panel_path = os.path.join(self.artifacts_dir, "panel.json")
        
        result = subprocess.run([
            sys.executable, self.script_path,
            "--reports-glob", os.path.join(self.reports_dir, "*.json"),
            "--output-panel", panel_path
        ], capture_output=True, text=True)

        self.assertEqual(result.returncode, 0)
        self.assertTrue(os.path.exists(panel_path))
        with open(panel_path, 'r') as f:
            panel = json.load(f)
        self.assertEqual(panel['status_light'], 'GREEN')
        self.assertIn("OK", panel['headline'])

    def test_script_block_status(self):
        """Test the script with a report that should result in a BLOCK status."""
        create_mock_report_file(self.reports_dir, datetime.now(), critical=1)
        
        panel_path = os.path.join(self.artifacts_dir, "panel.json")

        result = subprocess.run([
            sys.executable, self.script_path,
            "--reports-glob", os.path.join(self.reports_dir, "*.json"),
            "--output-panel", panel_path
        ], capture_output=True, text=True)

        self.assertEqual(result.returncode, 1, "Script should exit 1 on BLOCK status")
        with open(panel_path, 'r') as f:
            panel = json.load(f)
        self.assertEqual(panel['status_light'], 'RED')

    def test_script_warn_status(self):
        """Test the script with reports that should result in a WARN status."""
        now = datetime.now()
        create_mock_report_file(self.reports_dir, now - timedelta(days=1), warning=1)
        create_mock_report_file(self.reports_dir, now, warning=3) # Degrading trend
        
        panel_path = os.path.join(self.artifacts_dir, "panel.json")

        result = subprocess.run([
            sys.executable, self.script_path,
            "--reports-glob", os.path.join(self.reports_dir, "*.json"),
            "--output-panel", panel_path
        ], capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0, "Script should exit 0 on WARN status")
        with open(panel_path, 'r') as f:
            panel = json.load(f)
        self.assertEqual(panel['status_light'], 'AMBER')

    def test_history_file_append(self):
        """Test that the --append-history flag creates and appends to the history file."""
        history_path = os.path.join(self.artifacts_dir, "history.jsonl")
        
        # Run 1
        create_mock_report_file(self.reports_dir, datetime.now(), warning=1)
        subprocess.run([
            sys.executable, self.script_path,
            "--reports-glob", os.path.join(self.reports_dir, "*.json"),
            "--history-file", history_path, "--append-history"
        ], check=True)
        
        self.assertTrue(os.path.exists(history_path))
        with open(history_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)

        # Run 2
        create_mock_report_file(self.reports_dir, datetime.now() + timedelta(seconds=1), critical=1)
        subprocess.run([
            sys.executable, self.script_path,
            "--reports-glob", os.path.join(self.reports_dir, "*.json"),
            "--history-file", history_path, "--append-history"
        ], check=False) # check=False because it will exit 1

        with open(history_path, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 2)
        last_entry = json.loads(lines[-1])
        self.assertEqual(last_entry['status'], 'BLOCK')
        self.assertEqual(last_entry['critical_count'], 1)

if __name__ == '__main__':
    unittest.main(verbosity=2)
