# backend/governance_signal/tests/test_canonical.py
import json
import unittest
from backend.governance_signal.canonical import to_canonical_json

class TestCanonicalization(unittest.TestCase):

    def test_canonicalization_is_deterministic(self):
        """
        Tests that the canonicalization function is deterministic,
        producing the same output for logically identical but structurally
        different inputs.
        """
        signal_base = {
            "signalId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "originatorId": "test-origin",
            "timestamp": "2025-12-10T22:00:00Z",
            "semanticType": "HEARTBEAT_OK",
            "severity": 1,
            "ttl": 60,
            "cryptographicMetadata": {
                "signature": "should_be_removed",
                "publicKey": "pubkey",
                "signingAlgorithm": "ECDSA-P256"
            }
        }
        
        # Signal with keys in a different order
        signal_reordered = {
            "originatorId": "test-origin",
            "signalId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
            "severity": 1,
            "cryptographicMetadata": {
                "signingAlgorithm": "ECDSA-P256",
                "publicKey": "pubkey",
                "signature": "should_be_removed"
            },
            "ttl": 60,
            "timestamp": "2025-12-10T22:00:00Z",
            "semanticType": "HEARTBEAT_OK"
        }
        
        canonical1 = to_canonical_json(signal_base)
        canonical2 = to_canonical_json(signal_reordered)
        
        self.assertEqual(canonical1, canonical2, "Canonical JSON should be identical regardless of key order.")

    def test_signature_is_removed(self):
        """
        Tests that the 'signature' field is correctly removed from the
        cryptographicMetadata during canonicalization.
        """
        signal = {
            "signalId": "some-id",
            "cryptographicMetadata": {
                "signature": "this_must_be_removed",
                "publicKey": "pubkey"
            }
        }
        
        canonical_json = to_canonical_json(signal)
        reloaded_data = json.loads(canonical_json)
        
        self.assertNotIn("signature", reloaded_data["cryptographicMetadata"])
        self.assertIn("publicKey", reloaded_data["cryptographicMetadata"])

if __name__ == "__main__":
    unittest.main()
