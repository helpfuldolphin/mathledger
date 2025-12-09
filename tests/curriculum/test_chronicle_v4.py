# PHASE IV — NOT USED IN PHASE I
# File: tests/curriculum/test_chronicle_v4.py
import unittest
import os
import json
import shutil
import sys
from datetime import datetime, timedelta

EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'experiments'))
sys.path.insert(0, EXPERIMENTS_DIR)

from curriculum_linter_v3 import (
    build_curriculum_drift_chronicle,
    evaluate_curriculum_for_promotion,
    build_curriculum_director_panel
)

def create_mock_report(timestamp, critical=0, warning=0, info=0, critical_paths=None):
    """Helper to create a mock drift report."""
    if critical_paths is None: critical_paths = []
    drifts = []
    for i in range(critical): drifts.append({'severity': 'CRITICAL', 'parameter': critical_paths[i % len(critical_paths)] if critical_paths else f'param.crit{i}'})
    for i in range(warning): drifts.append({'severity': 'WARNING', 'parameter': f'param.warn{i}'})
    for i in range(info): drifts.append({'severity': 'INFO', 'parameter': f'param.info{i}'})
    return {
        "report_generated_utc": timestamp.isoformat(),
        "summary": {"severity_counts": {"CRITICAL": critical, "WARNING": warning, "INFO": info}},
        "drifts": drifts
    }

class TestChronicleV4(unittest.TestCase):
    def test_chronicle_generation(self):
        """TASK 1: Test chronicle generation, including trend and recurrent paths."""
        now = datetime.now()
        reports = [
            create_mock_report(now - timedelta(days=2), critical=1, warning=2, critical_paths=['path.A']),
            create_mock_report(now - timedelta(days=1), critical=1, warning=1, critical_paths=['path.A']),
            create_mock_report(now, warning=3)
        ]
        chronicle = build_curriculum_drift_chronicle(reports)
        self.assertEqual(chronicle['schema_version'], "4.0")
        self.assertEqual(len(chronicle['drift_events_series']), 3)
        self.assertEqual(chronicle['recurrent_drift_paths'], ['path.A'])
        self.assertEqual(chronicle['drift_trend'], 'DEGRADING')

    def test_chronicle_improving_trend(self):
        """TASK 1: Test 'IMPROVING' trend detection."""
        now = datetime.now()
        reports = [create_mock_report(now - timedelta(days=1), warning=5), create_mock_report(now, warning=2)]
        chronicle = build_curriculum_drift_chronicle(reports)
        self.assertEqual(chronicle['drift_trend'], 'IMPROVING')

    def test_promotion_coupler(self):
        """TASK 2: Test the promotion coupler logic for BLOCK, WARN, and OK."""
        now = datetime.now()
        latest_report_crit = create_mock_report(now, critical=1)
        chronicle_stable = build_curriculum_drift_chronicle([])
        eval_crit = evaluate_curriculum_for_promotion(latest_report_crit, chronicle_stable)
        self.assertEqual(eval_crit['status'], 'BLOCK')

        latest_report_warn = create_mock_report(now, warning=5)
        chronicle_degrading = {'drift_trend': 'DEGRADING'}
        eval_warn = evaluate_curriculum_for_promotion(latest_report_warn, chronicle_degrading)
        self.assertEqual(eval_warn['status'], 'WARN')

        latest_report_ok = create_mock_report(now, info=1)
        eval_ok = evaluate_curriculum_for_promotion(latest_report_ok, chronicle_stable)
        self.assertEqual(eval_ok['status'], 'OK')

    def test_director_panel_generation(self):
        """TASK 3: Test generation of the director panel view with the corrected assertion."""
        now = datetime.now()
        latest_report = create_mock_report(now, critical=1, critical_paths=['parameters.depth_max'])
        chronicle = build_curriculum_drift_chronicle([latest_report])
        promotion_eval = evaluate_curriculum_for_promotion(latest_report, chronicle)
        
        panel = build_curriculum_director_panel(latest_report, promotion_eval, chronicle)
        self.assertEqual(panel['status_light'], 'RED')
        self.assertEqual(panel['drift_severity'], 'CRITICAL')
        self.assertIn("BLOCK", panel['headline']) # Corrected, more robust assertion

if __name__ == '__main__':
    unittest.main(verbosity=2)
