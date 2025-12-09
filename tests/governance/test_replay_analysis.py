# PHASE IV — CROSS-RUN GOVERNANCE RADAR & DETERMINISM CONTRACT ENFORCER
"""
Tests for the Replay Governance Radar, Promotion Coupler, and Director Panel.
"""
import unittest
import os
import yaml
from backend.governance.replay_analysis import (
    build_replay_governance_radar,
    evaluate_replay_for_promotion,
    build_replay_director_panel,
)

# --- Synthetic Data for different states ---

STABLE_LEDGER = {"total_receipts": 100, "number_verified": 98, "number_failed": 2}
STABLE_INCIDENTS = [{"status": "FAILED", "recon_codes_seen": ["RECON-002"]}]
STABLE_SUMMARY = {"status": "OK"}

DRIFTING_LEDGER = {"total_receipts": 100, "number_verified": 94, "number_failed": 6}
DRIFTING_INCIDENTS = [
    {"status": "FAILED", "recon_codes_seen": ["RECON-002"]},
    {"status": "FAILED", "recon_codes_seen": ["RECON-003"]},
]
DRIFTING_SUMMARY = {"status": "WARN"}

UNSTABLE_LEDGER = {"total_receipts": 100, "number_verified": 89, "number_failed": 11}
UNSTABLE_INCIDENTS = [
    {"status": "FAILED", "recon_codes_seen": ["RECON-001", "RECON-005"]},
    {"status": "FAILED", "recon_codes_seen": ["RECON-003"]},
    {"status": "FAILED", "recon_codes_seen": ["RECON-001", "RECON-005"]}, # Recurring critical
]
UNSTABLE_SUMMARY = {"status": "BLOCKED"}


class TestReplayAnalysis(unittest.TestCase):

    def test_radar_stable(self):
        """Verify radar reports STABLE correctly."""
        radar = build_replay_governance_radar(STABLE_LEDGER, STABLE_INCIDENTS)
        self.assertEqual(radar["radar_status"], "STABLE")
        self.assertEqual(len(radar["recurring_critical_incident_fingerprints"]), 0)

    def test_radar_drifting(self):
        """Verify radar reports DRIFTING correctly."""
        radar = build_replay_governance_radar(DRIFTING_LEDGER, DRIFTING_INCIDENTS)
        self.assertEqual(radar["radar_status"], "DRIFTING")

    def test_radar_unstable(self):
        """Verify radar reports UNSTABLE on low rate."""
        radar = build_replay_governance_radar(UNSTABLE_LEDGER, [])
        self.assertEqual(radar["radar_status"], "UNSTABLE")
        
    def test_radar_unstable_on_critical_incidents(self):
        """Verify radar reports UNSTABLE on recurring critical incidents."""
        # High determinism rate, but with critical incidents
        ledger = {"total_receipts": 100, "number_verified": 99, "number_failed": 1}
        radar = build_replay_governance_radar(ledger, UNSTABLE_INCIDENTS)
        self.assertEqual(radar["radar_status"], "UNSTABLE")
        self.assertIn("FAILED:RECON-001:RECON-005", radar["recurring_critical_incident_fingerprints"])

    def test_promotion_coupler_ok(self):
        """Verify promotion coupler returns OK."""
        radar = build_replay_governance_radar(STABLE_LEDGER, STABLE_INCIDENTS)
        eval = evaluate_replay_for_promotion(radar, STABLE_SUMMARY)
        self.assertEqual(eval["status"], "OK")
        self.assertTrue(eval["replay_promotion_ok"])

    def test_promotion_coupler_warn(self):
        """Verify promotion coupler returns WARN."""
        radar = build_replay_governance_radar(DRIFTING_LEDGER, DRIFTING_INCIDENTS)
        eval = evaluate_replay_for_promotion(radar, DRIFTING_SUMMARY)
        self.assertEqual(eval["status"], "WARN")
        self.assertFalse(eval["replay_promotion_ok"])

    def test_promotion_coupler_block(self):
        """Verify promotion coupler returns BLOCK."""
        radar = build_replay_governance_radar(UNSTABLE_LEDGER, UNSTABLE_INCIDENTS)
        eval = evaluate_replay_for_promotion(radar, UNSTABLE_SUMMARY)
        self.assertEqual(eval["status"], "BLOCK")
        self.assertFalse(eval["replay_promotion_ok"])
        self.assertIn("Recurring critical incidents detected.", eval["reasons"])

    def test_director_panel_green(self):
        """Verify director panel shows GREEN."""
        radar = build_replay_governance_radar(STABLE_LEDGER, STABLE_INCIDENTS)
        promo_eval = evaluate_replay_for_promotion(radar, STABLE_SUMMARY)
        panel = build_replay_director_panel(radar, promo_eval)
        
        self.assertEqual(panel["status_light"], "GREEN")
        self.assertEqual(panel["determinism_rate"], "98.00%")
        self.assertIn("stable", panel["headline"])

    def test_director_panel_yellow(self):
        """Verify director panel shows YELLOW."""
        radar = build_replay_governance_radar(DRIFTING_LEDGER, DRIFTING_INCIDENTS)
        promo_eval = evaluate_replay_for_promotion(radar, DRIFTING_SUMMARY)
        panel = build_replay_director_panel(radar, promo_eval)
        
        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertEqual(panel["determinism_rate"], "94.00%")
        self.assertIn("drift", panel["headline"])
        
    def test_director_panel_red(self):
        """Verify director panel shows RED."""
        radar = build_replay_governance_radar(UNSTABLE_LEDGER, UNSTABLE_INCIDENTS)
        promo_eval = evaluate_replay_for_promotion(radar, UNSTABLE_SUMMARY)
        panel = build_replay_director_panel(radar, promo_eval)
        
        self.assertEqual(panel["status_light"], "RED")
        self.assertEqual(panel["determinism_rate"], "89.00%")
        self.assertIn("unstable", panel["headline"])
        self.assertEqual(panel["critical_incident_count"], 1)

    def test_criticality_config_loading(self):
        """Verify that changing the config file changes the radar outcome."""
        # Create a temp config file that defines a non-default code as critical
        temp_config_content = {
            "critical_recon_codes": ["RECON-002"]
        }
        temp_config_path = "test_criticality_rules.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(temp_config_content, f)

        # This incident is normally not critical
        incidents = [
            {"status": "FAILED", "recon_codes_seen": ["RECON-002"]},
            {"status": "FAILED", "recon_codes_seen": ["RECON-002"]},
        ]
        ledger = {"total_receipts": 100, "number_verified": 98, "number_failed": 2}

        # Run radar with the custom config
        radar = build_replay_governance_radar(ledger, incidents, criticality_config_path=temp_config_path)

        # Expect it to be UNSTABLE because RECON-002 is now considered critical and recurring
        self.assertEqual(radar["radar_status"], "UNSTABLE")
        self.assertIn("FAILED:RECON-002", radar["recurring_critical_incident_fingerprints"])

        os.remove(temp_config_path)


if __name__ == "__main__":
    unittest.main()