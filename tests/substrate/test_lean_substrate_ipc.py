"""
Unit tests for the Lean Substrate IPC interface, specifically the canonical
serialization and hashing functions.
"""
import unittest
from backend.substrate.lean_substrate_interface import (
    SubstrateRequest,
    SubstrateBudget,
    _canonical_json_dumps,
    compute_request_hash,
    compute_determinism_hash
)

class TestLeanSubstrateIPC(unittest.TestCase):

    def test_canonical_json_dumps(self):
        """
        Ensures serialization is deterministic: sorted keys, no whitespace,
        and correct UTF-8 encoding.
        """
        data_1 = {"b": 1, "a": 2, "c": {"z": 9, "x": 8}}
        data_2 = {"a": 2, "c": {"x": 8, "z": 9}, "b": 1} # Same data, different order

        # Unicode characters should not be escaped.
        data_unicode = {"name": "Gödel"}
        expected_unicode = b'{"name":"G\xc3\xb6del"}'

        self.assertEqual(
            _canonical_json_dumps(data_1),
            _canonical_json_dumps(data_2)
        )
        self.assertEqual(
            _canonical_json_dumps(data_1),
            b'{"a":2,"b":1,"c":{"x":8,"z":9}}'
        )
        self.assertEqual(
            _canonical_json_dumps(data_unicode),
            expected_unicode
        )

    def test_compute_request_hash(self):
        """
        Verifies that the request hash is stable and correct.
        """
        budget = SubstrateBudget(cycle_budget_s=10.0, taut_timeout_s=12.0)
        request = SubstrateRequest(
            protocol_version="1.0",
            item_id="test_item_123",
            cycle_seed=42,
            formula="theorem T : true := trivial",
            budget=budget
        )
        # This expected hash is pre-calculated from the canonical JSON form.
        expected_hash = "7a13915104d4128a8d0530b427b2f913d317f2a1b0288863f53857e3198e09f5"
        self.assertEqual(compute_request_hash(request), expected_hash)

    def test_compute_determinism_hash(self):
        """
        Verifies that the determinism hash for a result payload is stable.
        """
        payload = {"proven": True, "steps": ["step1", "step2"]}
        # This expected hash is pre-calculated.
        expected_hash = "92004d33a0058e578f72295697a5116e4bb691238697b5a86efb83478b8a6a66"
        self.assertEqual(compute_determinism_hash(payload), expected_hash)

if __name__ == '__main__':
    unittest.main()
