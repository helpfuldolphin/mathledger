# PHASE II — NOT USED IN PHASE I
#
# Unit tests for the Substrate Abstraction Layer.

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the experiments directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.substrate.substrate import (FoSubstrate, MockSubstrate,
                                       SubstrateResult)

class TestMockSubstrate(unittest.TestCase):
    
    def test_execute_success_arithmetic(self):
        """Test successful execution of an arithmetic item."""
        print("PHASE II — NOT USED IN PHASE I: Running TestMockSubstrate.test_execute_success_arithmetic")
        substrate = MockSubstrate()
        item = "2 + 2"
        seed = 123
        result = substrate.execute(item, seed)
        
        self.assertIsInstance(result, SubstrateResult)
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.result_data.get("mock_result"), 4)
        self.assertEqual(len(result.verified_hashes), 1)

    def test_execute_success_algebra(self):
        """Test successful execution of an algebra-like item."""
        print("PHASE II — NOT USED IN PHASE I: Running TestMockSubstrate.test_execute_success_algebra")
        substrate = MockSubstrate()
        item = "algebra_expansion"
        seed = 456
        result = substrate.execute(item, seed)

        self.assertTrue(result.success)
        self.assertEqual(result.result_data.get("mock_result"), "Expanded(algebra_expansion)")
        self.assertTrue(result.verified_hashes[0].startswith("mock_verified_"))

    def test_execute_failure(self):
        """Test handling of an item that causes an exception."""
        print("PHASE II — NOT USED IN PHASE I: Running TestMockSubstrate.test_execute_failure")
        substrate = MockSubstrate()
        item = "invalid syntax"
        seed = 789
        result = substrate.execute(item, seed)
        
        self.assertIsInstance(result, SubstrateResult)
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        self.assertIn("MockSubstrate eval failed", result.error)
        self.assertEqual(result.verified_hashes, [])


class TestFoSubstrate(unittest.TestCase):

    @patch('subprocess.run')
    def test_execute_success(self, mock_subprocess_run):
        """Test successful execution by mocking the subprocess call."""
        print("PHASE II — NOT USED IN PHASE I: Running TestFoSubstrate.test_execute_success")
        
        # Configure the mock to return a successful process result
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_output = {
            "outcome": "VERIFIED",
            "verified_hashes": ["fo_verified_abc"],
            "log": "Test log"
        }
        mock_process.stdout = json.dumps(mock_output)
        mock_process.stderr = ""
        mock_subprocess_run.return_value = mock_process
        
        substrate = FoSubstrate(fo_script_path="dummy_script.py")
        item = "test_item"
        seed = 123
        result = substrate.execute(item, seed)
        
        # Verify the command that was called
        expected_command = [
            sys.executable,
            "dummy_script.py",
            "--item",
            "test_item",
            "--seed",
            "123"
        ]
        mock_subprocess_run.assert_called_once_with(
            expected_command,
            capture_output=True,
            text=True,
            check=True,
            timeout=30
        )
        
        # Verify the result object
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(result.result_data, mock_output)
        self.assertEqual(result.verified_hashes, ["fo_verified_abc"])

    @patch('subprocess.run')
    def test_execute_script_error(self, mock_subprocess_run):
        """Test handling of a script failure (non-zero exit code)."""
        print("PHASE II — NOT USED IN PHASE I: Running TestFoSubstrate.test_execute_script_error")
        
        # Configure the mock to simulate a failed process
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="test_cmd",
            stderr="A script error occurred."
        )
        
        substrate = FoSubstrate()
        result = substrate.execute("test_item", 123)
        
        self.assertFalse(result.success)
        self.assertIn("failed with exit code 1", result.error)
        self.assertIn("A script error occurred", result.error)

    @patch('subprocess.run')
    def test_execute_json_decode_error(self, mock_subprocess_run):
        """Test handling of invalid JSON output from the script."""
        print("PHASE II — NOT USED IN PHASE I: Running TestFoSubstrate.test_execute_json_decode_error")

        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "This is not valid JSON."
        mock_subprocess_run.return_value = mock_process

        substrate = FoSubstrate()
        result = substrate.execute("test_item", 123)

        self.assertFalse(result.success)
        self.assertIn("failed to decode JSON", result.error)
        self.assertIn("This is not valid JSON", result.error)


if __name__ == '__main__':
    unittest.main()
