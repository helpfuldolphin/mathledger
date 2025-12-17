"""Fusion test for GGFL integration."""
import unittest
from backend.ggfl.adapter import create_ggfl_contribution

class TestGgflIntegration(unittest.TestCase):

    def test_risk_band_mapping(self):
        """Tests the direct mapping from risk band to GGFL severity."""
        
        # Scenario: Low risk, nominal metrics
        contribution = create_ggfl_contribution("LOW", 0.01, 50.0)
        self.assertEqual(contribution["severity"], "INFO")
        self.assertIn("Baseline", contribution["details"]["justification"])

        # Scenario: Medium risk, nominal metrics
        contribution = create_ggfl_contribution("MEDIUM", 0.01, 50.0)
        self.assertEqual(contribution["severity"], "WARNING")
        self.assertIn("Baseline", contribution["details"]["justification"])

    def test_fusion_escalation_on_divergence(self):
        """Tests that severity is escalated when high risk and high divergence interact."""
        
        # Scenario: High risk, but divergence is low
        contribution = create_ggfl_contribution("HIGH", 0.01, 50.0)
        self.assertEqual(contribution["severity"], "ERROR") # Baseline for HIGH
        
        # Scenario: High risk AND high divergence -> CRITICAL
        contribution = create_ggfl_contribution("HIGH", 0.02, 50.0)
        self.assertEqual(contribution["severity"], "CRITICAL")
        self.assertIn("divergence", contribution["details"]["justification"])

    def test_fusion_escalation_on_budget(self):
        """Tests that severity is escalated when medium risk and low budget interact."""
        
        # Scenario: Medium risk, but budget is plentiful
        contribution = create_ggfl_contribution("MEDIUM", 0.01, 50.0)
        self.assertEqual(contribution["severity"], "WARNING") # Baseline for MEDIUM
        
        # Scenario: Medium risk AND low budget -> ERROR
        contribution = create_ggfl_contribution("MEDIUM", 0.01, 5.0)
        self.assertEqual(contribution["severity"], "ERROR")
        self.assertIn("budget", contribution["details"]["justification"])

    def test_critical_risk_remains_critical(self):
        """Ensures that a CRITICAL risk band always results in a CRITICAL severity."""
        
        contribution = create_ggfl_contribution("CRITICAL", 0.01, 50.0)
        self.assertEqual(contribution["severity"], "CRITICAL")


if __name__ == '__main__':
    unittest.main()
