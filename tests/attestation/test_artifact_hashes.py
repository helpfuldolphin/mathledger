"""
Tests for artifact hash computation and audit reporting.

PHASE II — NOT USED IN PHASE I
"""

import json
import tempfile
import unittest
from pathlib import Path

from experiments.manifest_verifier import (
    compute_artifact_hash,
    verify_artifact_hash,
    hash_string,
)
from experiments.audit_uplift_u2 import (
    audit_experiment,
    format_audit_report_json,
    format_audit_report_markdown,
    count_jsonl_lines,
)


class TestArtifactHashes(unittest.TestCase):
    """Test cryptographic hash computation."""
    
    def test_compute_artifact_hash_small_file(self):
        """Verify SHA-256 computation for a small temporary file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write('test content\n')
            tmp_path = Path(f.name)
        
        try:
            hash_value = compute_artifact_hash(tmp_path)
            
            # Verify hash is 64-character hex string
            self.assertEqual(len(hash_value), 64)
            self.assertTrue(all(c in '0123456789abcdef' for c in hash_value))
            
            # Verify determinism - same file produces same hash
            hash_value2 = compute_artifact_hash(tmp_path)
            self.assertEqual(hash_value, hash_value2)
            
            # Verify specific hash for known content
            # SHA-256 of literal string 'test content\n' (with actual newline)
            # This hash was computed independently to validate correctness
            expected = "a1fff0ffefb9eace7230c24e50731f0a91c62f9cefdfe77121c2f607125dffae"
            self.assertEqual(hash_value, expected)
        finally:
            tmp_path.unlink()
    
    def test_compute_artifact_hash_missing_file(self):
        """Missing files should return 'missing'."""
        nonexistent = Path("/tmp/nonexistent_file_12345.txt")
        hash_value = compute_artifact_hash(nonexistent)
        self.assertEqual(hash_value, "missing")
    
    def test_compute_artifact_hash_empty_file(self):
        """Empty file should produce specific hash."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write nothing
            tmp_path = Path(f.name)
        
        try:
            hash_value = compute_artifact_hash(tmp_path)
            
            # SHA-256 of empty string (well-known constant)
            # This is the standard SHA-256 hash of zero bytes
            expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            self.assertEqual(hash_value, expected)
        finally:
            tmp_path.unlink()
    
    def test_verify_artifact_hash(self):
        """Test hash verification."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write('verify me')
            tmp_path = Path(f.name)
        
        try:
            correct_hash = compute_artifact_hash(tmp_path)
            
            # Correct hash should verify
            self.assertTrue(verify_artifact_hash(tmp_path, correct_hash))
            
            # Wrong hash should not verify
            wrong_hash = "0" * 64
            self.assertFalse(verify_artifact_hash(tmp_path, wrong_hash))
        finally:
            tmp_path.unlink()
    
    def test_hash_string(self):
        """Test string hashing."""
        text = "hello world"
        hash_value = hash_string(text)
        
        self.assertEqual(len(hash_value), 64)
        
        # Verify determinism
        hash_value2 = hash_string(text)
        self.assertEqual(hash_value, hash_value2)
        
        # Different strings produce different hashes
        hash_value3 = hash_string("hello world!")
        self.assertNotEqual(hash_value, hash_value3)


class TestAuditReporting(unittest.TestCase):
    """Test audit report generation with embedded hashes."""
    
    def setUp(self):
        """Create temporary experiment directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.exp_dir = Path(self.temp_dir) / "test_experiment"
        self.exp_dir.mkdir()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_audit_with_valid_manifest(self):
        """Test audit of valid experiment with hash computation."""
        # Create manifest
        manifest = {
            "slice": "test_slice",
            "mode": "baseline",
            "cycles": 5,
            "outputs": {
                "results": "uplift_u2_test_slice_baseline.jsonl"
            }
        }
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Create log file
        log_path = self.exp_dir / "uplift_u2_test_slice_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"cycle": i}) + "\n")
        
        # Run audit
        result = audit_experiment(self.exp_dir)
        
        # Check status
        self.assertEqual(result.status, "PASS")
        self.assertEqual(len(result.findings), 0)
        
        # Check artifact hashes were computed
        self.assertIn("manifest.json", result.artifact_hashes)
        self.assertIn("uplift_u2_test_slice_baseline.jsonl", result.artifact_hashes)
        
        # Verify hashes are valid hex strings
        for artifact, hash_val in result.artifact_hashes.items():
            self.assertEqual(len(hash_val), 64)
            self.assertTrue(all(c in '0123456789abcdef' for c in hash_val))
    
    def test_json_report_contains_hashes(self):
        """Verify JSON report includes artifact_hashes block."""
        # Create minimal valid experiment
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 1
        }
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        log_path = self.exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            f.write(json.dumps({"cycle": 0}) + "\n")
        
        result = audit_experiment(self.exp_dir)
        json_report = format_audit_report_json(result)
        
        # Verify structure
        self.assertIn("artifact_hashes", json_report)
        self.assertIsInstance(json_report["artifact_hashes"], dict)
        self.assertGreater(len(json_report["artifact_hashes"]), 0)
    
    def test_markdown_report_contains_hash_table(self):
        """Verify Markdown report includes hash table."""
        # Create minimal valid experiment
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 1
        }
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        log_path = self.exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            f.write(json.dumps({"cycle": 0}) + "\n")
        
        result = audit_experiment(self.exp_dir)
        md_report = format_audit_report_markdown(result)
        
        # Verify hash section exists
        self.assertIn("## Artifact Hashes (SHA-256)", md_report)
        self.assertIn("| Artifact | Hash |", md_report)
        
        # Verify explanation text
        self.assertIn("tamper-evident", md_report.lower())
        self.assertIn("sha-256", md_report.lower())
    
    def test_audit_missing_manifest(self):
        """Test audit when manifest is missing."""
        # Empty directory
        result = audit_experiment(self.exp_dir)
        
        self.assertEqual(result.status, "MISSING")
        self.assertGreater(len(result.findings), 0)
        self.assertEqual(result.findings[0]["severity"], "MISSING")
        self.assertIn("Manifest", result.findings[0]["message"])
    
    def test_audit_empty_log_file(self):
        """Test detection of empty JSONL files."""
        # Create manifest
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 5
        }
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Create empty log file
        log_path = self.exp_dir / "uplift_u2_test_baseline.jsonl"
        log_path.touch()
        
        result = audit_experiment(self.exp_dir)
        
        self.assertEqual(result.status, "FAIL")
        error_findings = [f for f in result.findings if f["severity"] == "ERROR"]
        self.assertGreater(len(error_findings), 0)
        self.assertTrue(any("empty" in f["message"].lower() for f in error_findings))
    
    def test_count_jsonl_lines(self):
        """Test JSONL line counting utility."""
        # Create temp file with known lines
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            f.write('{"line": 1}\n')
            f.write('{"line": 2}\n')
            f.write('\n')  # Empty line should not count
            f.write('{"line": 3}\n')
            tmp_path = Path(f.name)
        
        try:
            count = count_jsonl_lines(tmp_path)
            self.assertEqual(count, 3)
        finally:
            tmp_path.unlink()
    
    def test_audit_cycle_count_mismatch(self):
        """Test detection of cycle count mismatch."""
        manifest = {
            "slice": "test",
            "mode": "baseline",
            "cycles": 10  # Declare 10 cycles
        }
        manifest_path = self.exp_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Create log with only 5 cycles
        log_path = self.exp_dir / "uplift_u2_test_baseline.jsonl"
        with open(log_path, 'w') as f:
            for i in range(5):
                f.write(json.dumps({"cycle": i}) + "\n")
        
        result = audit_experiment(self.exp_dir)
        
        self.assertEqual(result.status, "FAIL")
        mismatch_findings = [f for f in result.findings 
                            if f["category"] == "cycle_count"]
        self.assertGreater(len(mismatch_findings), 0)
        self.assertIn("mismatch", mismatch_findings[0]["message"].lower())


class TestDeterminism(unittest.TestCase):
    """Test deterministic behavior of hash computation."""
    
    def test_hash_computation_is_deterministic(self):
        """Same file content must produce same hash."""
        content = b"determinism test content"
        
        hashes = []
        for _ in range(10):
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                f.write(content)
                tmp_path = Path(f.name)
            
            try:
                hash_val = compute_artifact_hash(tmp_path)
                hashes.append(hash_val)
            finally:
                tmp_path.unlink()
        
        # All hashes must be identical
        self.assertTrue(all(h == hashes[0] for h in hashes))


if __name__ == '__main__':
    unittest.main()
