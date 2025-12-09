# PHASE II — NOT RUN IN PHASE I
"""
End-to-end test for the governance pipeline, from synthetic artifacts
to a final governance receipt.
"""
import unittest
import os
import json
import hashlib
import yaml
from backend.governance.receipt_builder import build_governance_receipt
from scripts.verify_prereg import verify_preregistration
from scripts.verify_manifest_integrity import verify_manifest
from scripts.verify_uplift_gates import verify_gates

class TestPipelineEndToEnd(unittest.TestCase):
    """
    Simulates a full, valid experiment run and verifies that it passes
    the entire governance pipeline and produces a correct receipt.
    """

    @classmethod
    def setUpClass(cls):
        """Create a complete, valid set of synthetic artifacts."""
        cls.run_dir = "tests/governance/test_data/valid_run"
        os.makedirs(cls.run_dir, exist_ok=True)

        # 1. Create synthetic raw files
        slice_content = '{"parameter": "alpha", "value": 0.1}'
        results_content = '{"steps": 100, "final_value": 0.987}'
        with open(os.path.join(cls.run_dir, "slice.json"), "w") as f: f.write(slice_content)
        with open(os.path.join(cls.run_dir, "results.jsonl"), "w") as f: f.write(results_content)

        # 2. Calculate their hashes
        slice_hash = hashlib.sha256(slice_content.encode()).hexdigest()
        cls.results_hash = hashlib.sha256(results_content.encode()).hexdigest()

        # 3. Create preregistration file using the slice hash
        prereg_entry = {
            "experiment_id": "U2_E2E_TEST_001",
            "description": "Full pipeline end-to-end test.",
            "slice_config": "slice.json",
            "slice_config_hash": slice_hash,
            "seed": 2025,
            "success_metrics": ["final_value"]
        }
        with open(os.path.join(cls.run_dir, "prereg.yaml"), "w") as f: yaml.dump([prereg_entry], f)

        # 4. Calculate prereg hash
        prereg_hash = hashlib.sha256(yaml.dump(prereg_entry, sort_keys=True).encode()).hexdigest()

        # 5. Create the manifest binding everything together
        cls.manifest_data = {
            "manifest_schema_version": "1.0",
            "experiment_id": "U2_E2E_TEST_001",
            "preregistration_hash": prereg_hash,
            "slice_config_hash": slice_hash,
            "results_hash": cls.results_hash,
            "code_version_hash": "e2e-test-commit-hash",
            "deterministic_seed": 2025
        }
        cls.manifest_path = os.path.join(cls.run_dir, "manifest.json")
        with open(cls.manifest_path, "w") as f: json.dump(cls.manifest_data, f)
        
        # 6. Define the expected receipt snapshot
        # Manually construct the expected output for assertion.
        from backend.governance.receipt_builder import generate_deterministic_timestamp
        cls.expected_receipt = {
            "receipt_version": "GOVERNANCE-1.0.0",
            "experiment_id": "U2_E2E_TEST_001",
            "manifest_path": cls.manifest_path,
            "verification_timestamp": generate_deterministic_timestamp(cls.results_hash),
            "governance_record": {
                "prereg_state": {"status": "VERIFIED", "preregistration_hash": prereg_hash},
                "manifest_state": {
                    "status": "VERIFIED",
                    "code_version_hash": "e2e-test-commit-hash",
                    "deterministic_seed": 2025
                },
                "hash_state": {
                    "status": "VERIFIED",
                    "verified_hashes": {"slice_config": slice_hash, "results": cls.results_hash}
                },
                "integrity_state": {
                    "status": "VERIFIED",
                    "verified_bindings": ["preregistration", "slice_config", "results"]
                },
                "final_decision": {
                    "decision": "admissible",
                    "message": "All Phase II governance gates passed."
                }
            },
            "$schema": "docs/U2_GOVERNANCE_PIPELINE.md#4-canonical-governance-receipt-json-schema"
        }


    @classmethod
    def tearDownClass(cls):
        """Clean up the synthetic artifacts."""
        for f in ["slice.json", "results.jsonl", "prereg.yaml", "manifest.json"]:
            os.remove(os.path.join(cls.run_dir, f))
        os.rmdir(cls.run_dir)
        os.rmdir(os.path.dirname(cls.run_dir))


    def test_pipeline_simulation(self):
        """
        Runs the full pipeline of verification scripts and the receipt builder
        and compares the final output to a golden snapshot.
        """
        # --- Simulate CI Pipeline ---

        # Step 1: Run prereg verification
        prereg_result = verify_preregistration(
            os.path.join(self.run_dir, "prereg.yaml"), "U2_E2E_TEST_001")
        self.assertEqual(prereg_result["exit_code"], 0)

        # Step 2: Run manifest integrity verification
        manifest_result = verify_manifest(
            self.manifest_path,
            os.path.join(self.run_dir, "prereg.yaml"),
            os.path.join(self.run_dir, "slice.json"),
            os.path.join(self.run_dir, "results.jsonl")
        )
        self.assertEqual(manifest_result["exit_code"], 0)
        
        # Step 3: Run uplift gates verification
        uplift_gate_result = verify_gates(self.manifest_path)
        self.assertEqual(uplift_gate_result["exit_code"], 0)
        
        # Step 4: Build the receipt
        actual_receipt = build_governance_receipt(
            prereg_result, manifest_result, uplift_gate_result, self.manifest_data
        )

        # Final Assertion: Compare the actual receipt to the expected snapshot
        self.assertDictEqual(actual_receipt, self.expected_receipt)

if __name__ == "__main__":
    unittest.main()
