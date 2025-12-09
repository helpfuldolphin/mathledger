# tests/telemetry/test_slo_evaluator.py
#
# UNIT TEST: Telemetry SLO Evaluator & Auto-Action Engine
# JURISDICTION: Telemetry Integrity, SLO Enforcement, Automated Governance
# IDENTITY: GEMINI H, Telemetry Sentinel

import unittest
import sys
import os

# Ensure backend modules can be imported
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from backend.telemetry.slo_evaluator import (
    evaluate_telemetry_slo,
    decide_telemetry_publication,
    summarize_telemetry_slo_for_global_health,
)

class TestTelemetrySLOEvaluator(unittest.TestCase):

    def setUp(self):
        """Define the canonical SLO configuration for all tests."""
        self.slo_config = {
            "max_quarantine_ratio": 0.10,      # 10%
            "warn_quarantine_ratio": 0.05,     #  5%
            "min_l2_percentage": 0.70,         # 70%
            "max_critical_violations": 0,
        }

    def test_slo_status_ok(self):
        """
        GIVEN: A telemetry snapshot well within all SLO thresholds.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be OK with no reasons.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 10,   # 1% ratio
            "l2_record_count": 800,      # 80% L2
            "critical_violations": [],
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "OK")
        self.assertEqual(len(result["reasons"]), 0)

    def test_slo_status_warn_on_quarantine_ratio(self):
        """
        GIVEN: A snapshot exceeding the WARN quarantine ratio but not the BREACH ratio.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be WARN with a corresponding reason.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 70,   # 7% ratio
            "l2_record_count": 800,      # 80% L2
            "critical_violations": [],
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "WARN")
        self.assertEqual(len(result["reasons"]), 1)
        self.assertIn("WARN: Quarantine ratio", result["reasons"][0])

    def test_slo_status_breach_on_quarantine_ratio(self):
        """
        GIVEN: A snapshot exceeding the BREACH quarantine ratio.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be BREACH.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 110,  # 11% ratio
            "l2_record_count": 800,
            "critical_violations": [],
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "BREACH")
        self.assertIn("BREACH: Quarantine ratio", result["reasons"][0])

    def test_slo_status_breach_on_l2_percentage(self):
        """
        GIVEN: A snapshot with an L2 record percentage below the SLO.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be BREACH.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 10,
            "l2_record_count": 650,      # 65% L2
            "critical_violations": [],
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "BREACH")
        self.assertIn("BREACH: L2 record percentage", result["reasons"][0])

    def test_slo_status_breach_on_critical_violations(self):
        """
        GIVEN: A snapshot containing critical violations.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be BREACH.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 10,
            "l2_record_count": 800,
            "critical_violations": [{"error": "TemporalOrderError", "index": 50}],
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "BREACH")
        self.assertIn("BREACH: Found 1 critical violations", result["reasons"][0])

    def test_slo_status_breach_multiple_reasons(self):
        """
        GIVEN: A snapshot violating multiple SLOs.
        WHEN: evaluate_telemetry_slo is called.
        THEN: The status must be BREACH and all reasons must be listed.
        """
        snapshot = {
            "total_records": 1000,
            "quarantined_records": 150,  # Breach
            "l2_record_count": 500,      # Breach
            "critical_violations": [{"error": "SchemaError"}], # Breach
        }
        
        result = evaluate_telemetry_slo(snapshot, self.slo_config)
        
        self.assertEqual(result["slo_status"], "BREACH")
        self.assertEqual(len(result["reasons"]), 3)

    def test_publication_decision_publish(self):
        """
        GIVEN: An SLO result of OK.
        WHEN: decide_telemetry_publication is called.
        THEN: The recommendation must be PUBLISH.
        """
        slo_result = {"slo_status": "OK", "reasons": []}
        decision = decide_telemetry_publication({}, slo_result)
        
        self.assertEqual(decision["recommended_action"], "PUBLISH")
        self.assertTrue(decision["publish_allowed"])
        self.assertFalse(decision["require_manual_review"])

    def test_publication_decision_warn(self):
        """
        GIVEN: An SLO result of WARN.
        WHEN: decide_telemetry_publication is called.
        THEN: The recommendation must be PUBLISH_WITH_WARNING.
        """
        slo_result = {"slo_status": "WARN", "reasons": ["..."]}
        decision = decide_telemetry_publication({}, slo_result)
        
        self.assertEqual(decision["recommended_action"], "PUBLISH_WITH_WARNING")
        self.assertTrue(decision["publish_allowed"])
        self.assertTrue(decision["require_manual_review"])

    def test_publication_decision_quarantine(self):
        """
        GIVEN: An SLO result of BREACH.
        WHEN: decide_telemetry_publication is called.
        THEN: The recommendation must be QUARANTINE.
        """
        slo_result = {"slo_status": "BREACH", "reasons": ["..."]}
        decision = decide_telemetry_publication({}, slo_result)
        
        self.assertEqual(decision["recommended_action"], "QUARANTINE")
        self.assertFalse(decision["publish_allowed"])
        self.assertTrue(decision["require_manual_review"])
        
    def test_global_health_summary(self):
        """
        GIVEN: Various SLO results.
        WHEN: summarize_telemetry_slo_for_global_health is called.
        THEN: It must produce the correct simplified summary.
        """
        # OK case
        slo_ok = {"slo_status": "OK", "reasons": []}
        summary_ok = summarize_telemetry_slo_for_global_health(slo_ok)
        self.assertEqual(summary_ok["slo_status"], "OK")
        self.assertFalse(summary_ok["any_breach"])
        self.assertEqual(summary_ok["key_reason"], "All SLOs met.")

        # Breach case
        reason_text = "BREACH: Quarantine ratio 11.00% exceeds SLO of 10.00%"
        slo_breach = {"slo_status": "BREACH", "reasons": [reason_text, "another reason"]}
        summary_breach = summarize_telemetry_slo_for_global_health(slo_breach)
        self.assertEqual(summary_breach["slo_status"], "BREACH")
        self.assertTrue(summary_breach["any_breach"])
        self.assertEqual(summary_breach["key_reason"], reason_text)

if __name__ == "__main__":
    unittest.main()
