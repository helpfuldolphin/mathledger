# backend/governance_signal/tests/test_validator.py
import json
import os
import unittest
from backend.governance_signal.validator import validate_signal

TEST_DIR = os.path.dirname(__file__)

class TestValidator(unittest.TestCase):

    def test_valid_signal_passes(self):
        """Tests that a valid signal passes schema validation."""
        valid_signal_path = os.path.join(TEST_DIR, "valid_signal.json")
        with open(valid_signal_path, "r") as f:
            signal_data = json.load(f)
        
        is_valid, message = validate_signal(signal_data)
        self.assertTrue(is_valid, f"Validation should pass, but failed with: {message}")
        self.assertEqual(message, "Signal is valid.")

    def test_invalid_signal_fails(self):
        """Tests that an invalid signal (missing 'ttl') fails schema validation."""
        invalid_signal_path = os.path.join(TEST_DIR, "invalid_signal.json")
        with open(invalid_signal_path, "r") as f:
            signal_data = json.load(f)
            
        is_valid, message = validate_signal(signal_data)
        self.assertFalse(is_valid, "Validation should fail, but it passed.")
        self.assertIn("'ttl' is a required property", message, "The error message did not indicate the correct missing field.")

    def test_invalid_field_type_fails(self):
        """Tests that a signal with a wrong data type for a field fails."""
        valid_signal_path = os.path.join(TEST_DIR, "valid_signal.json")
        with open(valid_signal_path, "r") as f:
            signal_data = json.load(f)
        
        # Corrupt the data type of 'severity'
        signal_data["severity"] = "should_be_an_integer"
        
        is_valid, message = validate_signal(signal_data)
        self.assertFalse(is_valid, "Validation should fail due to wrong type, but it passed.")
        self.assertIn("'should_be_an_integer' is not of type 'integer'", message)

if __name__ == "__main__":
    # You will need to install jsonschema to run these tests: pip install jsonschema
    unittest.main()
