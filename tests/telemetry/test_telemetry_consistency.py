# tests/telemetry/test_telemetry_consistency.py

import unittest
import sys
import os
import tempfile
import json
import subprocess
import datetime
from pathlib import Path

# Ensure modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

def generate_test_data(is_consistent=True, num_cycles=10):
    """Generates a set of test data files in a specified directory."""
    baseline_run_id = "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    rfl_run_id = "c4b9b7e0-4c7a-4d3b-9e6a-8d7d3e6f9f3c"
    
    cycle_events = []
    base_time = datetime.datetime.utcnow()

    # Baseline events (80% success)
    base_success_count = int(num_cycles * 0.8)
    for i in range(num_cycles):
        cycle_events.append({
            "ts": (base_time + datetime.timedelta(seconds=i)).isoformat() + "Z",
            "run_id": baseline_run_id, "slice": "U2_env_A", "mode": "baseline", "cycle": i,
            "success": i < base_success_count,
            "metric_type": "duration", "metric_value": 1.0, "H_t": "1"*64
        })

    # RFL events (90% success)
    rfl_success_count = int(num_cycles * 0.9)
    for i in range(num_cycles):
         cycle_events.append({
            "ts": (base_time + datetime.timedelta(seconds=num_cycles+i)).isoformat() + "Z",
            "run_id": rfl_run_id, "slice": "U2_env_A", "mode": "rfl", "cycle": i,
            "success": i < rfl_success_count,
            "metric_type": "duration", "metric_value": 1.0, "H_t": "2"*64
        })

    summary = {
        "slice": "U2_env_A", "mode": "rfl_v2_exp1",
        "n_cycles": {"baseline": num_cycles, "rfl": num_cycles},
        "p_base": base_success_count / num_cycles, 
        "p_rfl": rfl_success_count / num_cycles, 
        "delta": (rfl_success_count - base_success_count) / num_cycles,
        "CI": {"lower_bound": -0.05, "upper_bound": 0.11},
        "baseline_run_id": baseline_run_id, "rfl_run_id": rfl_run_id
    }
    
    if not is_consistent:
        # Introduce an inconsistency for testing
        summary["delta"] = 99.9 # Grossly incorrect delta
        cycle_events[5]["ts"] = cycle_events[6]["ts"] # Non-monotonic timestamp

    return cycle_events, summary

class TestTelemetryConsistencyEndToEnd(unittest.TestCase):

    def find_check_status(self, report, check_name):
        """Helper to find the status of a specific check in the report."""
        for check in report.get("checks", []):
            if check.get("check_name") == check_name:
                return check.get("status")
        return None

    def run_audit_script(self, temp_dir):
        """Executes the audit script as a subprocess."""
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_telemetry_audit.py"
        result = subprocess.run(
            [sys.executable, str(script_path), temp_dir],
            capture_output=True,
            text=True
        )
        report_path = Path(temp_dir) / "artifacts" / "telemetry" / "telemetry_consistency_report.json"
        
        report = None
        if report_path.exists():
            with open(report_path, 'r') as f:
                report = json.load(f)
        
        return result.returncode, report, result.stdout, result.stderr

    def test_end_to_end_consistent_data(self):
        """
        Tests the full pipeline with consistent data, expecting exit code 0.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            cycle_events, summary = generate_test_data(is_consistent=True)
            
            with open(Path(temp_dir) / "cycles.jsonl", "w") as f:
                for event in cycle_events:
                    f.write(json.dumps(event) + "\n")
            with open(Path(temp_dir) / "summary.json", "w") as f:
                json.dump(summary, f)

            exit_code, report, stdout, stderr = self.run_audit_script(temp_dir)

            self.assertEqual(exit_code, 0, f"Script failed unexpectedly. Stderr: {stderr}")
            self.assertIsNotNone(report)
            for check in report["checks"]:
                with self.subTest(check_name=check["check_name"]):
                    self.assertEqual(check["status"], "PASSED")

    def test_end_to_end_inconsistent_data(self):
        """
        Tests the full pipeline with inconsistent data, expecting exit code 1.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # This data introduces three errors: bad delta range, non-monotonic timestamp, and bad delta arithmetic
            cycle_events, summary = generate_test_data(is_consistent=False)

            with open(Path(temp_dir) / "cycles.jsonl", "w") as f:
                for event in cycle_events:
                    f.write(json.dumps(event) + "\n")
            with open(Path(temp_dir) / "summary.json", "w") as f:
                json.dump(summary, f)

            exit_code, report, stdout, stderr = self.run_audit_script(temp_dir)
            
            self.assertEqual(exit_code, 1, "Script should have failed but exited with 0.")
            self.assertIsNotNone(report)
            self.assertEqual(self.find_check_status(report, "value_ranges"), "FAILED", "Did not detect value range error for delta.")
            self.assertEqual(self.find_check_status(report, "monotonic_timestamps"), "FAILED", "Did not detect non-monotonic timestamp.")
            self.assertEqual(self.find_check_status(report, "delta_arithmetic"), "FAILED", "Did not detect delta arithmetic error.")

    def test_end_to_end_missing_file(self):
        """
        Tests the full pipeline with a missing data file, expecting exit code 2.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # We don't write any files
            exit_code, report, stdout, stderr = self.run_audit_script(temp_dir)
            
            self.assertEqual(exit_code, 2, "Script should have failed with file not found error.")
            self.assertIsNone(report)
            self.assertIn("Summary file not found", stderr)

if __name__ == '__main__':
    unittest.main()
