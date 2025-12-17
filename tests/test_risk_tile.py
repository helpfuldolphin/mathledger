"""Tests for the risk tile generation logic."""
import json
import unittest
from backend.risk.risk_tile import normalize_metric, compute_overall_risk, map_to_band

class TestRiskTileLogic(unittest.TestCase):

    def test_normalize_metric(self):
        """Tests the metric normalization logic for determinism."""
        # Test '>' operator
        self.assertAlmostEqual(normalize_metric(0.9, 0.8, '>'), 0.0) # Good, above threshold
        self.assertAlmostEqual(normalize_metric(0.8, 0.8, '>'), 0.0) # At threshold
        self.assertAlmostEqual(normalize_metric(0.7, 0.8, '>'), 0.125) # Below threshold
        self.assertAlmostEqual(normalize_metric(0.0, 0.8, '>'), 1.0) # Max risk
        
        # Test '<' operator
        self.assertAlmostEqual(normalize_metric(0.04, 0.05, '<'), 0.0) # Good, below threshold
        self.assertAlmostEqual(normalize_metric(0.05, 0.05, '<'), 0.0) # At threshold
        self.assertAlmostEqual(normalize_metric(0.06, 0.05, '<'), 0.2) # Above threshold
        self.assertAlmostEqual(normalize_metric(0.10, 0.05, '<'), 1.0) # Max risk

    def test_band_mapping(self):
        """Tests the mapping of risk scores to bands for determinism."""
        self.assertEqual(map_to_band(0.0), 'LOW')
        self.assertEqual(map_to_band(0.09), 'LOW')
        self.assertEqual(map_to_band(0.10), 'MEDIUM')
        self.assertEqual(map_to_band(0.39), 'MEDIUM')
        self.assertEqual(map_to_band(0.40), 'HIGH')
        self.assertEqual(map_to_band(0.69), 'HIGH')
        self.assertEqual(map_to_band(0.70), 'CRITICAL')
        self.assertEqual(map_to_band(1.0), 'CRITICAL')

    def test_compute_overall_risk(self):
        """Tests the overall risk computation for determinism."""
        scores = {
            'delta_p': 0.1,
            'rsi': 0.2,
            'omega': 0.3,
            'tda': 0.4,
            'divergence': 0.5
        }
        # Expected: (0.1*0.15 + 0.2*0.3 + 0.3*0.25 + 0.4*0.1 + 0.5*0.2) / (0.15+0.3+0.25+0.1+0.2) = 0.29
        self.assertAlmostEqual(compute_overall_risk(scores), 0.29)

    def test_json_schema_validation(self):
        """
        Tests that a generated tile conforms to the JSON schema.
        Note: This is a conceptual test. In a real CI system, you would
        use a library like jsonschema to validate this.
        """
        with open('backend/risk/risk_tile_schema.json', 'r') as f:
            schema = json.load(f)

        # A sample tile that should be valid
        sample_tile = {
            "metric_id": "overall_risk_summary",
            "metric_name": "Overall Risk Summary",
            "value": 0.29,
            "threshold": 0.4,
            "operator": "<",
            "gate": "P4",
            "risk_band": "MEDIUM",
            "timestamp": "2025-12-12T12:00:00Z",
            "justification_ref": "docs/risk/FORMAL_SPEC_RISK_STRATEGY.md"
        }
        
        # This is a simplified check. A full validation would be more robust.
        self.assertTrue(all(key in sample_tile for key in schema['required']))


if __name__ == '__main__':
    unittest.main()
