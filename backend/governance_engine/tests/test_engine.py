# backend/governance_engine/tests/test_engine.py
import unittest
import os
import yaml
from backend.governance_engine.engine import GovernanceEngine
from backend.governance_signal.model import GovernanceSignal, CryptographicMetadata

TEST_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(TEST_DIR, "../../.."))

def create_mock_signal(semanticType="HEARTBEAT_OK", severity=1):
    """Factory function to create mock signals for testing."""
    return GovernanceSignal("id", "origin", "ts", semanticType, severity, 60, CryptographicMetadata("s", "p", "a"), payload={})

class TestGovernanceEngine(unittest.TestCase):

    def setUp(self):
        """Set up the default policy engine for tests."""
        self.default_policy_path = os.path.join(PROJECT_ROOT, "governance_policies/default_policy.yaml")
        self.engine = GovernanceEngine(self.default_policy_path)
        self.assertTrue(self.engine.is_stable, "Engine should be stable with default policy and all dependencies.")

    def test_invalid_policy_causes_unstable_state(self):
        """Tests that a policy violating the schema makes the engine unstable."""
        invalid_policy_content = {"default_action": "LOG", "rules": [{"name": "Broke"}]} # Missing 'decision'
        temp_policy_path = os.path.join(TEST_DIR, "invalid_temp_policy.yaml")
        with open(temp_policy_path, "w") as f: yaml.dump(invalid_policy_content, f)
        
        unstable_engine = GovernanceEngine(temp_policy_path)
        self.assertFalse(unstable_engine.is_stable)
        decision = unstable_engine.decide(create_mock_signal())
        self.assertEqual(decision.action, "UNSTABLE_POLICY")

        os.remove(temp_policy_path)

    def test_golden_default_policy_high_severity_alert(self):
        """Golden Test: Verifies a key deterministic behavior of the default policy."""
        signal = create_mock_signal(semanticType="RESOURCE_CONSTRAINT_WARN", severity=4)
        decision = self.engine.decide(signal)
        self.assertEqual(decision.action, "TRIGGER_ALERT")
        self.assertEqual(decision.triggering_rule, "MEDIUM: Behavioral Anomalies and Resource Warnings")

    def test_missing_pyyaml_causes_unstable_state(self):
        """Tests that a missing pyyaml library makes the engine unstable."""
        from backend.governance_engine import engine
        original_yaml = engine.yaml
        try:
            engine.yaml = None
            engine_without_deps = GovernanceEngine(self.default_policy_path)
            self.assertFalse(engine_without_deps.is_stable)
            decision = engine_without_deps.decide(create_mock_signal())
            self.assertEqual(decision.action, "UNSTABLE_POLICY")
        finally:
            engine.yaml = original_yaml

    def test_missing_jsonschema_causes_unstable_state(self):
        """Tests that a missing jsonschema library makes the engine unstable."""
        from backend.governance_engine import engine
        original_jsonschema = engine.jsonschema
        try:
            engine.jsonschema = None
            engine_without_deps = GovernanceEngine(self.default_policy_path)
            self.assertFalse(engine_without_deps.is_stable)
            decision = engine_without_deps.decide(create_mock_signal())
            self.assertEqual(decision.action, "UNSTABLE_POLICY")
        finally:
            engine.jsonschema = original_jsonschema

if __name__ == "__main__":
    unittest.main()