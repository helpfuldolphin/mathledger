# tests/test_manifest_verifier.py
"""
PHASE II — NOT RUN IN PHASE I

Unit tests for U2 manifest cryptographic binding verification.

These tests verify:
1. SHA-256 computation functions
2. Preregistration binding verification
3. Slice config binding verification
4. Hₜ-series integrity verification
5. Report generation

This module performs ZERO uplift interpretation.
"""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.manifest_verifier import (
    PHASE_LABEL,
    ManifestVerifier,
    compute_sha256_canonical_json,
    compute_sha256_file,
    compute_sha256_string,
    verify_manifest,
)


class TestSHA256Functions(unittest.TestCase):
    """Test SHA-256 computation functions."""

    def test_compute_sha256_string_deterministic(self):
        """SHA-256 of same string should always be identical."""
        test_string = "Hello, World!"
        hash1 = compute_sha256_string(test_string)
        hash2 = compute_sha256_string(test_string)
        self.assertEqual(hash1, hash2)
        # Verify against known hash
        expected = hashlib.sha256(test_string.encode('utf-8')).hexdigest()
        self.assertEqual(hash1, expected)

    def test_compute_sha256_string_different_inputs(self):
        """Different strings should produce different hashes."""
        hash1 = compute_sha256_string("input1")
        hash2 = compute_sha256_string("input2")
        self.assertNotEqual(hash1, hash2)

    def test_compute_sha256_file(self):
        """SHA-256 of file should match content hash."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            f.flush()
            temp_path = Path(f.name)
        
        try:
            file_hash = compute_sha256_file(temp_path)
            content_hash = compute_sha256_string("Test file content")
            self.assertEqual(file_hash, content_hash)
        finally:
            temp_path.unlink()

    def test_compute_sha256_canonical_json_deterministic(self):
        """Canonical JSON hashing should be deterministic regardless of key order."""
        obj1 = {"b": 2, "a": 1, "c": 3}
        obj2 = {"a": 1, "c": 3, "b": 2}  # Same data, different order
        
        hash1 = compute_sha256_canonical_json(obj1)
        hash2 = compute_sha256_canonical_json(obj2)
        
        self.assertEqual(hash1, hash2)

    def test_compute_sha256_canonical_json_nested(self):
        """Canonical JSON hashing should work with nested objects."""
        obj = {
            "outer": {
                "inner": {"z": 26, "a": 1},
                "list": [3, 2, 1],
            },
            "value": "test",
        }
        
        # Should not raise and should be deterministic
        hash1 = compute_sha256_canonical_json(obj)
        hash2 = compute_sha256_canonical_json(obj)
        self.assertEqual(hash1, hash2)


class TestManifestVerifier(unittest.TestCase):
    """Test ManifestVerifier class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test files
        self.prereg_content = {
            "preregistration": {
                "experiment_family": "uplift_u2",
                "version": 1,
                "status": "NOT_YET_EXECUTED",
            }
        }
        self.prereg_path = self.temp_path / "prereg.yaml"
        with open(self.prereg_path, 'w') as f:
            yaml.dump(self.prereg_content, f)
        
        self.config_content = {
            "version": "2.0",
            "slices": {
                "test_slice": {
                    "description": "Test slice",
                    "items": ["item1", "item2"],
                }
            }
        }
        self.config_path = self.temp_path / "config.yaml"
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config_content, f)
        
        # Compute expected hashes
        self.prereg_hash = compute_sha256_file(self.prereg_path)
        self.slice_config_hash = compute_sha256_canonical_json(
            self.config_content["slices"]["test_slice"]
        )
        
        # Create test log records
        self.log_records = [
            {"cycle": 0, "item": "item1", "success": True},
            {"cycle": 1, "item": "item2", "success": False},
        ]
        self.ht_series_hash = compute_sha256_string(
            json.dumps(self.log_records, sort_keys=True)
        )
        
        self.logs_path = self.temp_path / "test.jsonl"
        with open(self.logs_path, 'w') as f:
            for record in self.log_records:
                f.write(json.dumps(record) + "\n")
        
        # Create manifest with correct hashes
        self.manifest_content = {
            "label": PHASE_LABEL,
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 2,
            "prereg_hash": self.prereg_hash,
            "slice_config_hash": self.slice_config_hash,
            "ht_series_hash": self.ht_series_hash,
            "outputs": {
                "results": str(self.logs_path),
            }
        }
        self.manifest_path = self.temp_path / "manifest.json"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_verifier_initialization(self):
        """ManifestVerifier should initialize correctly."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        self.assertEqual(verifier.phase_label, PHASE_LABEL)
        self.assertEqual(verifier.verdict, "PENDING")

    def test_verify_prereg_binding_pass(self):
        """Preregistration binding should pass with correct hash."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_prereg_binding()
        self.assertTrue(result)
        
        # Check finding was recorded
        prereg_findings = [f for f in verifier.findings if f["check"] == "prereg_binding"]
        self.assertEqual(len(prereg_findings), 1)
        self.assertEqual(prereg_findings[0]["status"], "PASS")

    def test_verify_prereg_binding_fail(self):
        """Preregistration binding should fail with wrong hash."""
        # Modify manifest to have wrong prereg hash
        self.manifest_content["prereg_hash"] = "wrong_hash"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)
        
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_prereg_binding()
        self.assertFalse(result)
        
        prereg_findings = [f for f in verifier.findings if f["check"] == "prereg_binding"]
        self.assertEqual(prereg_findings[0]["status"], "FAIL")

    def test_verify_slice_config_binding_pass(self):
        """Slice config binding should pass with correct hash."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_slice_config_binding()
        self.assertTrue(result)

    def test_verify_slice_config_binding_fail(self):
        """Slice config binding should fail with wrong hash."""
        self.manifest_content["slice_config_hash"] = "wrong_hash"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)
        
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_slice_config_binding()
        self.assertFalse(result)

    def test_verify_ht_series_integrity_pass(self):
        """Hₜ-series integrity should pass with correct hash."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_ht_series_integrity(logs_path=self.logs_path)
        self.assertTrue(result)

    def test_verify_ht_series_integrity_fail(self):
        """Hₜ-series integrity should fail with wrong hash."""
        self.manifest_content["ht_series_hash"] = "wrong_hash"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)
        
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.load_artifacts()
        result = verifier.verify_ht_series_integrity(logs_path=self.logs_path)
        self.assertFalse(result)

    def test_verify_all_pass(self):
        """verify_all should return PASS when all checks pass."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verdict = verifier.verify_all(logs_path=self.logs_path)
        self.assertEqual(verdict, "PASS")

    def test_verify_all_fail_on_any_failure(self):
        """verify_all should return FAIL if any check fails."""
        self.manifest_content["prereg_hash"] = "wrong_hash"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)
        
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verdict = verifier.verify_all(logs_path=self.logs_path)
        self.assertEqual(verdict, "FAIL")

    def test_generate_json_report(self):
        """JSON report should contain all required fields."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.verify_all(logs_path=self.logs_path)
        report = verifier.generate_json_report()
        
        self.assertIn("label", report)
        self.assertEqual(report["label"], PHASE_LABEL)
        self.assertIn("verdict", report)
        self.assertIn("findings", report)
        self.assertIn("summary", report)
        self.assertIn("total_checks", report["summary"])

    def test_generate_markdown_report(self):
        """Markdown report should contain Phase II label."""
        verifier = ManifestVerifier(
            manifest_path=self.manifest_path,
            prereg_path=self.prereg_path,
            config_path=self.config_path,
        )
        verifier.verify_all(logs_path=self.logs_path)
        md_report = verifier.generate_markdown_report()
        
        self.assertIn(PHASE_LABEL, md_report)
        self.assertIn("Verdict", md_report)
        self.assertIn("Verification Results", md_report)


class TestVerifyManifestFunction(unittest.TestCase):
    """Test the verify_manifest convenience function."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create minimal manifest
        self.manifest_content = {
            "label": PHASE_LABEL,
            "slice": "test",
            "mode": "baseline",
            "cycles": 1,
        }
        self.manifest_path = self.temp_path / "manifest.json"
        with open(self.manifest_path, 'w') as f:
            json.dump(self.manifest_content, f)

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_verify_manifest_returns_tuple(self):
        """verify_manifest should return (verdict, json_report) tuple."""
        verdict, report = verify_manifest(manifest_path=self.manifest_path)
        
        self.assertIsInstance(verdict, str)
        self.assertIn(verdict, ["PASS", "FAIL"])
        self.assertIsInstance(report, dict)

    def test_verify_manifest_writes_json_report(self):
        """verify_manifest should write JSON report when path provided."""
        output_json = self.temp_path / "report.json"
        
        verify_manifest(
            manifest_path=self.manifest_path,
            output_json=output_json,
        )
        
        self.assertTrue(output_json.exists())
        with open(output_json) as f:
            report = json.load(f)
        self.assertIn("label", report)

    def test_verify_manifest_writes_markdown_report(self):
        """verify_manifest should write Markdown report when path provided."""
        output_md = self.temp_path / "report.md"
        
        verify_manifest(
            manifest_path=self.manifest_path,
            output_md=output_md,
        )
        
        self.assertTrue(output_md.exists())
        with open(output_md) as f:
            content = f.read()
        self.assertIn(PHASE_LABEL, content)


class TestPhaseLabel(unittest.TestCase):
    """Test that PHASE II label is correctly applied."""

    def test_phase_label_constant(self):
        """PHASE_LABEL should contain required text."""
        self.assertIn("PHASE II", PHASE_LABEL)
        self.assertIn("NOT", PHASE_LABEL)
        self.assertIn("PHASE I", PHASE_LABEL)


if __name__ == "__main__":
    unittest.main()
