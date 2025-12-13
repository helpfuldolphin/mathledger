"""Integration test for GGFL fusion with risk signal."""
import unittest
from backend.governance.fusion import build_global_alignment_view

class TestGgflFusionIntegration(unittest.TestCase):

    def test_precedence_invariant_with_hard_block(self):
        """
        Proves: A HARD_BLOCK from a high-precedence signal (Identity)
        is the final decision, regardless of the risk signal.
        """
        # A high-precedence signal causing a HARD_BLOCK
        identity_signal = {"block_hash_valid": False}

        # A low-threat risk signal
        risk_signal = {"risk_band": "LOW"}

        # Run fusion
        result = build_global_alignment_view(
            identity=identity_signal,
            risk=risk_signal,
        )

        # Assert that the HARD_BLOCK from Identity takes precedence
        self.assertEqual(result["fusion_result"]["decision"], "BLOCK")
        self.assertTrue(result["fusion_result"]["is_hard"])
        self.assertEqual(result["fusion_result"]["determining_signal"], "identity")
        self.assertEqual(result["escalation"]["level_name"], "L5_EMERGENCY")

    def test_shadow_mode_enforcement(self):
        """
        Proves: A CRITICAL risk signal does not change an ALLOW decision
        to BLOCK, confirming it is non-enforceable (shadow mode).
        """
        # Baseline signals that result in an ALLOW
        budget_signal = {"compute_budget_remaining": 0.8}
        topology_signal = {"within_omega": True}

        # First, confirm the baseline is ALLOW
        baseline_result = build_global_alignment_view(
            budget=budget_signal,
            topology=topology_signal,
        )
        self.assertEqual(baseline_result["fusion_result"]["decision"], "ALLOW")
        self.assertEqual(baseline_result["escalation"]["level_name"], "L0_NOMINAL")

        # Now, add a CRITICAL risk signal
        risk_signal = {"risk_band": "CRITICAL"}
        with_risk_result = build_global_alignment_view(
            budget=budget_signal,
            topology=topology_signal,
            risk=risk_signal,
        )
        
        # The final decision must remain ALLOW, proving non-enforcement
        self.assertEqual(with_risk_result["fusion_result"]["decision"], "ALLOW",
                         "Risk signal in shadow mode should not change final decision")
        self.assertFalse(with_risk_result["fusion_result"]["is_hard"])

        # However, the risk *recommendation* should be a soft BLOCK
        risk_recs = [r for r in with_risk_result["recommendations"] if r["signal_id"] == "risk"]
        self.assertTrue(any(rec["action"] == "BLOCK" for rec in risk_recs))
        
        # And the escalation level should be elevated to reflect the underlying risk
        self.assertEqual(with_risk_result["escalation"]["level_name"], "L2_DEGRADED",
                         "Escalation level should rise even if decision is unchanged")

if __name__ == '__main__':
    unittest.main()