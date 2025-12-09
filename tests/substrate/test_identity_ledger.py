# PHASE II — NOT USED IN PHASE I
#
# Conformance Test Suite for the Substrate Identity Ledger system.
# This suite validates Operation Substrate-SEAL.

import hashlib
import json
import os
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure the project root is in the path for backend imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.substrate.identity import SubstrateIdentityEnvelope, get_source_file_hash
from backend.substrate.substrate import Substrate, SubstrateResult

# --- Dummy Substrates for Testing the Identity System ---

class GoodSealedSubstrate(Substrate):
    """A substrate that is fully conformant and will be wrapped by the sealer."""
    def _execute_internal(self, item: str, cycle_seed: int) -> SubstrateResult:
        # Uses a deterministic, seeded PRNG as required by the spec.
        rng = random.Random(cycle_seed)
        val = rng.random()
        return SubstrateResult(
            success=True,
            result_data={"value": val, "item": item},
            verified_hashes=[f"good_{val:.5f}"]
        )

class BadFileSealedSubstrate(Substrate):
    """A substrate that violates the file I/O contract."""
    def _execute_internal(self, item: str, cycle_seed: int) -> SubstrateResult:
        # This write will be caught by the patched 'open' in the base class.
        with open("bad_file.tmp", "w") as f:
            f.write("violation")
        return SubstrateResult(success=True, result_data={}, verified_hashes=[])

# --- Identity Ledger Test Class ---

class TestIdentityLedger(unittest.TestCase):

    def tearDown(self):
        """Clean up any files created by bad substrates."""
        if os.path.exists("bad_file.tmp"):
            os.remove("bad_file.tmp")
            
    def test_T_ID_1_schema_conformance(self):
        """T-ID-1: Envelope from a good substrate must match the schema."""
        print("PHASE II [T-ID-1]: Testing Envelope Schema Conformance...")
        substrate = GoodSealedSubstrate()
        envelope_dict = substrate.execute("test_item", 123)
        
        # Check for top-level keys
        expected_keys = [
            "substrate_name", "version_hash", "spec_version",
            "execution_input", "execution_output", "forbidden_behavior_audit",
            "determinism_signature"
        ]
        self.assertCountEqual(envelope_dict.keys(), expected_keys)
        
        # Check nested keys
        self.assertIn("item", envelope_dict["execution_input"])
        self.assertIn("success", envelope_dict["execution_output"])
        self.assertIn("global_rng_check", envelope_dict["forbidden_behavior_audit"])
        
        # Check values
        self.assertEqual(envelope_dict["substrate_name"], "GoodSealedSubstrate")
        self.assertEqual(envelope_dict["execution_input"]["item"], "test_item")
        print("  -> PASSED")

    def test_T_ID_2_signature_reproducibility(self):
        """T-ID-2: Identical inputs must produce identical envelopes and signatures."""
        print("PHASE II [T-ID-2]: Testing Signature Reproducibility...")
        substrate = GoodSealedSubstrate()
        
        envelope1 = substrate.execute("item_A", 456)
        envelope2 = substrate.execute("item_A", 456)
        envelope3 = substrate.execute("item_B", 456) # Different item
        
        # Identical inputs must yield identical envelopes
        self.assertEqual(envelope1, envelope2)
        
        # Different inputs must yield different envelopes
        self.assertNotEqual(envelope1, envelope3)
        
        # Explicitly check the signature
        self.assertEqual(envelope1["determinism_signature"], envelope2["determinism_signature"])
        self.assertNotEqual(envelope1["determinism_signature"], envelope3["determinism_signature"])
        print("  -> PASSED")

    def test_T_ID_3_tamper_detection(self):
        """T-ID-3: A one-bit change to the envelope must invalidate the signature."""
        print("PHASE II [T-ID-3]: Testing Tamper Detection...")
        substrate = GoodSealedSubstrate()
        envelope_dict = substrate.execute("test_tamper", 789)
        original_signature = envelope_dict["determinism_signature"]
        
        # Tamper with the output data
        envelope_dict["execution_output"]["result_data"]["value"] += 0.001
        
        # Re-construct the envelope dataclass to use its verification logic
        tampered_envelope = SubstrateIdentityEnvelope(**envelope_dict)
        
        # The signature should now be invalid
        self.assertFalse(tampered_envelope.verify_signature())
        self.assertNotEqual(original_signature, tampered_envelope.determinism_signature)
        print("  -> PASSED")

    def test_T_ID_4_audit_capture(self):
        """T-ID-4: Side-effects in a bad substrate must be caught and logged in the audit."""
        print("PHASE II [T-ID-4]: Testing Audit Capture...")
        substrate = BadFileSealedSubstrate()
        envelope = substrate.execute("test_audit", 999)
        
        audit_report = envelope["forbidden_behavior_audit"]
        
        self.assertEqual(audit_report["file_io_check"], "FAILED")
        self.assertEqual(audit_report["global_rng_check"], "PASSED") # Should not have triggered this
        
        # Verify the signature is still valid, proving the audit result is part of the signed data
        envelope_obj = SubstrateIdentityEnvelope(**envelope)
        self.assertTrue(envelope_obj.verify_signature())
        print("  -> PASSED")

if __name__ == '__main__':
    unittest.main()
