# PHASE II — NOT USED IN PHASE I
#
# Unit tests for the Substrate Conformance Suite itself.
# This file tests the tester.

import os
import random
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure the backend directory is in the path for imports
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.substrate.conformance_suite import ConformanceTester, parse_spec_errors
from backend.substrate.substrate import Substrate, SubstrateResult

# --- Dummy Substrates for Testing ---

class GoodSubstrate(Substrate):
    """A substrate that is fully conformant."""
    def execute(self, item: str, cycle_seed: int) -> SubstrateResult:
        rng = random.Random(cycle_seed)
        val = rng.random()
        return SubstrateResult(
            success=True,
            result_data={"value": val},
            verified_hashes=[f"good_{val:.5f}"]
        )

class NondeterministicSubstrate(Substrate):
    """A substrate that is not deterministic."""
    def execute(self, item: str, cycle_seed: int) -> SubstrateResult:
        # This uses global state and will produce a different result on each call
        val = random.random()
        return SubstrateResult(
            success=True,
            result_data={"value": val}
        )

class FileWriterSubstrate(Substrate):
    """A substrate that writes a forbidden file."""
    def execute(self, item: str, cycle_seed: int) -> SubstrateResult:
        Path("forbidden_file.tmp").write_text("I should not exist.")
        return SubstrateResult(success=True, result_data={})

class GlobalRNGSubstrate(Substrate):
    """A substrate that uses the global random.random()."""
    def execute(self, item: str, cycle_seed: int) -> SubstrateResult:
        # The conformance suite patches the global random, so this will be detected.
        _ = random.random()
        return SubstrateResult(success=True, result_data={})

# --- Tests for the Conformance Suite ---

class TestConformanceSuite(unittest.TestCase):

    def setUp(self):
        """Parse the spec for error codes before running tests."""
        self.spec_errors = parse_spec_errors(Path(project_root) / "docs/U2_SUBSTRATE_SPECIFICATION.md")

    def tearDown(self):
        """Clean up any files created by bad substrates."""
        if os.path.exists("forbidden_file.tmp"):
            os.remove("forbidden_file.tmp")

    def test_good_substrate(self):
        """The suite should PASS a conformant substrate."""
        print("PHASE II: Testing Conformance Suite against GoodSubstrate")
        tester = ConformanceTester(GoodSubstrate, self.spec_errors)
        report = tester.run_suite()

        self.assertTrue(report.overall_passed())
        self.assertTrue(report.determinism_verdict.passed)
        
        behavior_passed = all(res.passed for res in report.behavioral_violations)
        self.assertTrue(behavior_passed, "GoodSubstrate failed a behavioral test.")

    def test_nondeterministic_substrate(self):
        """The suite should FAIL a non-deterministic substrate."""
        print("PHASE II: Testing Conformance Suite against NondeterministicSubstrate")
        tester = ConformanceTester(NondeterministicSubstrate, self.spec_errors)
        report = tester.run_suite()

        self.assertFalse(report.overall_passed())
        self.assertFalse(report.determinism_verdict.passed)
        self.assertIn("Failed", report.determinism_verdict.message)

    def test_file_writer_substrate(self):
        """The suite should FAIL a substrate that writes files."""
        print("PHASE II: Testing Conformance Suite against FileWriterSubstrate")
        # Need to import builtins to patch it correctly in the test context
        import builtins
        tester = ConformanceTester(FileWriterSubstrate, self.spec_errors)
        
        # The test_forbidden_behaviors method in the suite does the patching.
        # We just need to run it and check the result.
        behavior_results = tester.test_forbidden_behaviors()
        
        file_write_result = next(r for r in behavior_results if "file writes" in r.message)
        self.assertFalse(file_write_result.passed)
        self.assertIn("forbidden_file.tmp", file_write_result.details.get("files", []))
    
    def test_global_rng_substrate(self):
        """The suite should FAIL a substrate that uses global RNG."""
        print("PHASE II: Testing Conformance Suite against GlobalRNGSubstrate")
        tester = ConformanceTester(GlobalRNGSubstrate, self.spec_errors)
        
        behavior_results = tester.test_forbidden_behaviors()
        
        rng_result = next(r for r in behavior_results if "global random" in r.message)
        self.assertFalse(rng_result.passed)


if __name__ == '__main__':
    unittest.main()
