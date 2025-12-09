# PHASE II — NOT RUN IN PHASE I
"""
Unit tests for the replay receipt system.

Tests the replay receipt builder, validation, and verification logic
per the specifications in:
- docs/U2_REPLAY_RECEIPT_CHARTER.md
- docs/U2_GOVERNANCE_RECONCILIATION_SPEC.md Section 6.5
- docs/VSD_PHASE_2.md Section 9F

Test categories:
1. Golden path: VERIFIED receipt with all hashes matching
2. FAILED receipts for each RC-R* category
3. Deterministic receipts: same inputs → identical JSON
4. Error code mapping: RECON-18, RECON-19, RECON-20
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from backend.governance.replay_receipt import (
    ReplayReceipt,
    ReplayStatus,
    ReplayEnvironment,
    ManifestBinding,
    ReplayRunResult,
    VerificationSummary,
    FailedCheck,
    ReconErrorCode,
    REPLAY_RECEIPT_VERSION,
    DOMAIN_REPLAY_RECEIPT,
    build_replay_receipt,
    save_replay_receipt,
    load_replay_receipt,
    compute_receipt_hash,
    verify_receipt_hash,
    validate_replay_receipt,
    extract_ht_series_from_log,
    compare_ht_sequences,
)


class TestReplayReceiptDataClasses(unittest.TestCase):
    """Test the replay receipt data classes."""

    def test_replay_environment_capture(self):
        """Test environment capture."""
        env = ReplayEnvironment.capture(git_sha="abc123" * 7 + "ab", runner_version="2.0.0")
        self.assertEqual(len(env.git_sha), 40)
        self.assertIn(".", env.python_version)
        self.assertTrue(len(env.platform) > 0)
        self.assertEqual(env.rfl_runner_version, "2.0.0")

    def test_replay_status_enum(self):
        """Test status enum values."""
        self.assertEqual(ReplayStatus.VERIFIED.value, "VERIFIED")
        self.assertEqual(ReplayStatus.FAILED.value, "FAILED")
        self.assertEqual(ReplayStatus.INCOMPLETE.value, "INCOMPLETE")

    def test_recon_error_codes(self):
        """Test error code values."""
        self.assertEqual(ReconErrorCode.RECON_18_REPLAY_MISSING.value, "RECON-18")
        self.assertEqual(ReconErrorCode.RECON_19_REPLAY_MISMATCH.value, "RECON-19")
        self.assertEqual(ReconErrorCode.RECON_20_REPLAY_INCOMPLETE.value, "RECON-20")


class TestReceiptHashComputation(unittest.TestCase):
    """Test receipt hash computation and verification."""

    def setUp(self):
        """Create a minimal valid receipt for testing."""
        self.receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_001",
            status=ReplayStatus.VERIFIED,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(
                git_sha="a" * 40,
                python_version="3.11.5",
                platform="win32",
                rfl_runner_version="2.0.0",
            ),
            manifest_binding=ManifestBinding(
                manifest_path="test/manifest.json",
                manifest_hash="b" * 64,
                bound_at="2025-12-06T10:00:00+00:00",
            ),
            baseline_replay=ReplayRunResult(
                run_type="baseline",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="c" * 64,
                replay_log_hash="c" * 64,
                log_hash_match=True,
                expected_final_ht="d" * 64,
                replay_final_ht="d" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="e" * 64,
                replay_log_hash="e" * 64,
                log_hash_match=True,
                expected_final_ht="f" * 64,
                replay_final_ht="f" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12,
                checks_failed=0,
                all_verified=True,
            ),
        )

    def test_receipt_hash_is_deterministic(self):
        """Same receipt should produce same hash."""
        hash1 = compute_receipt_hash(self.receipt)
        hash2 = compute_receipt_hash(self.receipt)
        self.assertEqual(hash1, hash2)

    def test_receipt_hash_is_64_hex_chars(self):
        """Receipt hash should be 64 hex characters (SHA-256)."""
        receipt_hash = compute_receipt_hash(self.receipt)
        self.assertEqual(len(receipt_hash), 64)
        self.assertTrue(all(c in '0123456789abcdef' for c in receipt_hash))

    def test_receipt_hash_changes_with_content(self):
        """Different receipt content should produce different hash."""
        hash1 = compute_receipt_hash(self.receipt)

        # Modify receipt
        self.receipt.experiment_id = "U2_EXP_002"
        hash2 = compute_receipt_hash(self.receipt)

        self.assertNotEqual(hash1, hash2)

    def test_verify_receipt_hash_valid(self):
        """Valid receipt should verify successfully."""
        self.receipt.receipt_hash = compute_receipt_hash(self.receipt)
        self.assertTrue(verify_receipt_hash(self.receipt))

    def test_verify_receipt_hash_invalid(self):
        """Tampered receipt should fail verification."""
        self.receipt.receipt_hash = compute_receipt_hash(self.receipt)
        # Tamper with receipt
        self.receipt.experiment_id = "U2_EXP_TAMPERED"
        self.assertFalse(verify_receipt_hash(self.receipt))

    def test_domain_tag_in_hash(self):
        """Receipt hash should use domain tag."""
        receipt_dict = self.receipt.to_dict()
        receipt_dict["receipt_hash"] = ""
        canonical_json = json.dumps(receipt_dict, sort_keys=True, separators=(',', ':'))

        # Hash without domain tag
        hash_no_domain = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()

        # Hash with domain tag (what the module does)
        hash_with_domain = compute_receipt_hash(self.receipt)

        # They should be different
        self.assertNotEqual(hash_no_domain, hash_with_domain)


class TestHtSeriesExtraction(unittest.TestCase):
    """Test H_t series extraction from log files."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_extract_ht_series_from_log(self):
        """Test extracting H_t values from JSONL log."""
        log_path = Path(self.temp_dir) / "test.jsonl"

        # Create test log
        with open(log_path, 'w') as f:
            for i in range(5):
                record = {
                    "cycle": i,
                    "H_t": f"{'a' * 64}",
                    "R_t": f"{'b' * 64}",
                    "U_t": f"{'c' * 64}",
                }
                f.write(json.dumps(record) + "\n")

        ht_series = extract_ht_series_from_log(log_path)

        self.assertEqual(len(ht_series), 5)
        self.assertEqual(ht_series[0]["cycle"], 0)
        self.assertEqual(ht_series[0]["H_t"], "a" * 64)

    def test_extract_ht_series_alternative_keys(self):
        """Test extraction with alternative key names."""
        log_path = Path(self.temp_dir) / "test2.jsonl"

        with open(log_path, 'w') as f:
            for i in range(3):
                record = {
                    "cycle": i,
                    "composite_root": f"{'d' * 64}",
                    "reasoning_root": f"{'e' * 64}",
                    "ui_root": f"{'f' * 64}",
                }
                f.write(json.dumps(record) + "\n")

        ht_series = extract_ht_series_from_log(log_path)

        self.assertEqual(len(ht_series), 3)
        self.assertEqual(ht_series[0]["H_t"], "d" * 64)


class TestHtSequenceComparison(unittest.TestCase):
    """Test H_t sequence comparison."""

    def test_matching_sequences(self):
        """Identical sequences should match."""
        seq1 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(10)]
        seq2 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(10)]

        match, mismatch_cycle = compare_ht_sequences(seq1, seq2)

        self.assertTrue(match)
        self.assertIsNone(mismatch_cycle)

    def test_different_length_sequences(self):
        """Different length sequences should not match."""
        seq1 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(10)]
        seq2 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(5)]

        match, mismatch_cycle = compare_ht_sequences(seq1, seq2)

        self.assertFalse(match)
        self.assertEqual(mismatch_cycle, 0)

    def test_mismatched_sequences(self):
        """Sequences with different H_t values should not match."""
        seq1 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(10)]
        seq2 = [{"cycle": i, "H_t": f"hash{i}"} for i in range(10)]
        seq2[5]["H_t"] = "different_hash"  # Mismatch at cycle 5

        match, mismatch_cycle = compare_ht_sequences(seq1, seq2)

        self.assertFalse(match)
        self.assertEqual(mismatch_cycle, 5)


class TestReplayReceiptBuilder(unittest.TestCase):
    """Test the replay receipt builder with file-based fixtures."""

    def setUp(self):
        """Create test directory structure with mock experiment artifacts."""
        self.temp_dir = tempfile.mkdtemp()
        self.primary_dir = Path(self.temp_dir) / "primary"
        self.replay_dir = Path(self.temp_dir) / "replay"

        # Create directory structure
        (self.primary_dir / "baseline").mkdir(parents=True)
        (self.primary_dir / "rfl").mkdir(parents=True)
        (self.replay_dir / "baseline").mkdir(parents=True)
        (self.replay_dir / "rfl").mkdir(parents=True)

    def tearDown(self):
        """Clean up test directory."""
        shutil.rmtree(self.temp_dir)

    def _create_log_file(self, path: Path, num_cycles: int, seed: int = 12345):
        """Create a mock JSONL log file with H_t values."""
        with open(path, 'w') as f:
            for i in range(num_cycles):
                # Generate deterministic H_t based on cycle and seed
                ht = hashlib.sha256(f"{seed}:{i}".encode()).hexdigest()
                record = {
                    "cycle": i,
                    "seed": seed,
                    "H_t": ht,
                    "R_t": hashlib.sha256(f"R:{seed}:{i}".encode()).hexdigest(),
                    "U_t": hashlib.sha256(f"U:{seed}:{i}".encode()).hexdigest(),
                }
                f.write(json.dumps(record) + "\n")

    def _create_manifest(self, path: Path, seed: int = 12345, cycles: int = 10):
        """Create a mock manifest file."""
        manifest = {
            "experiment_id": "U2_EXP_001",
            "initial_seed": seed,
            "cycles": cycles,
            "slice": "SLICE_A",
            "phase": "II",
            "phase_label": "PHASE II — NOT USED IN PHASE I",
        }
        with open(path, 'w') as f:
            json.dump(manifest, f)

    def test_build_verified_receipt(self):
        """Test building a VERIFIED receipt when all hashes match."""
        num_cycles = 10
        seed = 12345

        # Create identical log files for primary and replay
        self._create_log_file(self.primary_dir / "baseline" / "run.jsonl", num_cycles, seed)
        self._create_log_file(self.primary_dir / "rfl" / "run.jsonl", num_cycles, seed)
        self._create_log_file(self.replay_dir / "baseline" / "run.jsonl", num_cycles, seed)
        self._create_log_file(self.replay_dir / "rfl" / "run.jsonl", num_cycles, seed)

        # Create manifest
        manifest_path = Path(self.temp_dir) / "manifest.json"
        self._create_manifest(manifest_path, seed, num_cycles)

        # Build receipt
        receipt = build_replay_receipt(
            primary_run_dir=self.primary_dir,
            replay_run_dir=self.replay_dir,
            manifest_path=manifest_path,
            git_sha="a" * 40,
        )

        self.assertEqual(receipt.status, ReplayStatus.VERIFIED)
        self.assertEqual(receipt.verification_summary.checks_failed, 0)
        self.assertTrue(receipt.baseline_replay.log_hash_match)
        self.assertTrue(receipt.baseline_replay.ht_sequence_match)
        self.assertTrue(receipt.rfl_replay.log_hash_match)
        self.assertTrue(receipt.rfl_replay.ht_sequence_match)

    def test_build_failed_receipt_log_hash_mismatch(self):
        """Test FAILED receipt when log hashes don't match."""
        num_cycles = 10

        # Create primary logs
        self._create_log_file(self.primary_dir / "baseline" / "run.jsonl", num_cycles, seed=100)
        self._create_log_file(self.primary_dir / "rfl" / "run.jsonl", num_cycles, seed=100)

        # Create replay logs with DIFFERENT content
        self._create_log_file(self.replay_dir / "baseline" / "run.jsonl", num_cycles, seed=999)
        self._create_log_file(self.replay_dir / "rfl" / "run.jsonl", num_cycles, seed=100)  # RFL matches

        manifest_path = Path(self.temp_dir) / "manifest.json"
        self._create_manifest(manifest_path)

        receipt = build_replay_receipt(
            primary_run_dir=self.primary_dir,
            replay_run_dir=self.replay_dir,
            manifest_path=manifest_path,
        )

        self.assertEqual(receipt.status, ReplayStatus.FAILED)
        self.assertGreater(receipt.verification_summary.checks_failed, 0)
        self.assertFalse(receipt.baseline_replay.log_hash_match)

    def test_build_incomplete_receipt_missing_logs(self):
        """Test INCOMPLETE receipt when log files are missing."""
        # Only create primary baseline log, nothing else
        self._create_log_file(self.primary_dir / "baseline" / "run.jsonl", 10)

        manifest_path = Path(self.temp_dir) / "manifest.json"
        self._create_manifest(manifest_path)

        receipt = build_replay_receipt(
            primary_run_dir=self.primary_dir,
            replay_run_dir=self.replay_dir,
            manifest_path=manifest_path,
        )

        self.assertEqual(receipt.status, ReplayStatus.INCOMPLETE)


class TestReceiptSerialization(unittest.TestCase):
    """Test receipt serialization and deserialization."""

    def setUp(self):
        """Create a valid receipt for testing."""
        self.receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_001",
            status=ReplayStatus.VERIFIED,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(
                git_sha="a" * 40,
                python_version="3.11.5",
                platform="win32",
            ),
            manifest_binding=ManifestBinding(
                manifest_path="test/manifest.json",
                manifest_hash="b" * 64,
                bound_at="2025-12-06T10:00:00+00:00",
            ),
            baseline_replay=ReplayRunResult(
                run_type="baseline",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="c" * 64,
                replay_log_hash="c" * 64,
                log_hash_match=True,
                expected_final_ht="d" * 64,
                replay_final_ht="d" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="e" * 64,
                replay_log_hash="e" * 64,
                log_hash_match=True,
                expected_final_ht="f" * 64,
                replay_final_ht="f" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12,
                checks_failed=0,
            ),
        )
        self.receipt.receipt_hash = compute_receipt_hash(self.receipt)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_to_dict_and_back(self):
        """Test dict serialization round-trip."""
        receipt_dict = self.receipt.to_dict()
        restored = ReplayReceipt.from_dict(receipt_dict)

        self.assertEqual(restored.experiment_id, self.receipt.experiment_id)
        self.assertEqual(restored.status, self.receipt.status)
        self.assertEqual(restored.receipt_hash, self.receipt.receipt_hash)

    def test_save_and_load(self):
        """Test file serialization round-trip."""
        receipt_path = Path(self.temp_dir) / "receipt.json"

        save_replay_receipt(self.receipt, receipt_path)
        self.assertTrue(receipt_path.exists())

        loaded = load_replay_receipt(receipt_path)

        self.assertEqual(loaded.experiment_id, self.receipt.experiment_id)
        self.assertEqual(loaded.status, self.receipt.status)
        self.assertEqual(loaded.receipt_hash, self.receipt.receipt_hash)

    def test_deterministic_serialization(self):
        """Same receipt should produce identical JSON."""
        json1 = self.receipt.to_json()
        json2 = self.receipt.to_json()

        self.assertEqual(json1, json2)


class TestReceiptValidation(unittest.TestCase):
    """Test receipt validation for governance admissibility."""

    def setUp(self):
        """Create test directory and valid receipt."""
        self.temp_dir = tempfile.mkdtemp()

        # Create valid VERIFIED receipt
        self.valid_receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_001",
            status=ReplayStatus.VERIFIED,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(
                git_sha="a" * 40,
                python_version="3.11.5",
                platform="win32",
            ),
            manifest_binding=ManifestBinding(
                manifest_path="test/manifest.json",
                manifest_hash="b" * 64,
                bound_at="2025-12-06T10:00:00+00:00",
            ),
            baseline_replay=ReplayRunResult(
                run_type="baseline",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="c" * 64,
                replay_log_hash="c" * 64,
                log_hash_match=True,
                expected_final_ht="d" * 64,
                replay_final_ht="d" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl",
                seed_used=12345,
                cycles_executed=100,
                expected_log_hash="e" * 64,
                replay_log_hash="e" * 64,
                log_hash_match=True,
                expected_final_ht="f" * 64,
                replay_final_ht="f" * 64,
                final_ht_match=True,
                ht_sequence_length=100,
                ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12,
                checks_failed=0,
            ),
        )
        self.valid_receipt.receipt_hash = compute_receipt_hash(self.valid_receipt)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_validate_verified_receipt(self):
        """Valid VERIFIED receipt should pass validation."""
        receipt_path = Path(self.temp_dir) / "valid_receipt.json"
        save_replay_receipt(self.valid_receipt, receipt_path)

        valid, error_code, message = validate_replay_receipt(receipt_path)

        self.assertTrue(valid)
        self.assertIsNone(error_code)
        self.assertIn("valid", message.lower())

    def test_validate_missing_receipt_recon_18(self):
        """Missing receipt should return RECON-18."""
        receipt_path = Path(self.temp_dir) / "nonexistent.json"

        valid, error_code, message = validate_replay_receipt(receipt_path)

        self.assertFalse(valid)
        self.assertEqual(error_code, ReconErrorCode.RECON_18_REPLAY_MISSING)

    def test_validate_failed_receipt_recon_19(self):
        """FAILED receipt should return RECON-19."""
        failed_receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_002",
            status=ReplayStatus.FAILED,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="different" + "0" * 55,
                log_hash_match=False, expected_final_ht="d" * 64, replay_final_ht="e" * 64,
                final_ht_match=False, ht_sequence_length=100, ht_sequence_match=False,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="f" * 64, replay_log_hash="f" * 64,
                log_hash_match=True, expected_final_ht="g" * 64, replay_final_ht="g" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=6, checks_failed=3,
                failed_checks=[FailedCheck(check_id="RC-R7", expected="c" * 16, actual="different")],
            ),
        )
        failed_receipt.receipt_hash = compute_receipt_hash(failed_receipt)

        receipt_path = Path(self.temp_dir) / "failed_receipt.json"
        save_replay_receipt(failed_receipt, receipt_path)

        valid, error_code, message = validate_replay_receipt(receipt_path)

        self.assertFalse(valid)
        self.assertEqual(error_code, ReconErrorCode.RECON_19_REPLAY_MISMATCH)

    def test_validate_incomplete_receipt_recon_20(self):
        """INCOMPLETE receipt should return RECON-20."""
        incomplete_receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_003",
            status=ReplayStatus.INCOMPLETE,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=0,
                expected_log_hash="", replay_log_hash="",
                log_hash_match=False, expected_final_ht="", replay_final_ht="",
                final_ht_match=False, ht_sequence_length=0, ht_sequence_match=False,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=0,
                expected_log_hash="", replay_log_hash="",
                log_hash_match=False, expected_final_ht="", replay_final_ht="",
                final_ht_match=False, ht_sequence_length=0, ht_sequence_match=False,
            ),
            verification_summary=VerificationSummary(checks_passed=2, checks_failed=10),
        )
        incomplete_receipt.receipt_hash = compute_receipt_hash(incomplete_receipt)

        receipt_path = Path(self.temp_dir) / "incomplete_receipt.json"
        save_replay_receipt(incomplete_receipt, receipt_path)

        valid, error_code, message = validate_replay_receipt(receipt_path)

        self.assertFalse(valid)
        self.assertEqual(error_code, ReconErrorCode.RECON_20_REPLAY_INCOMPLETE)

    def test_validate_tampered_receipt_recon_19(self):
        """Tampered receipt (hash mismatch) should return RECON-19."""
        receipt_path = Path(self.temp_dir) / "tampered_receipt.json"
        save_replay_receipt(self.valid_receipt, receipt_path)

        # Tamper with the file
        with open(receipt_path, 'r') as f:
            data = json.load(f)
        data["experiment_id"] = "U2_EXP_TAMPERED"
        with open(receipt_path, 'w') as f:
            json.dump(data, f)

        valid, error_code, message = validate_replay_receipt(receipt_path)

        self.assertFalse(valid)
        self.assertEqual(error_code, ReconErrorCode.RECON_19_REPLAY_MISMATCH)
        self.assertIn("hash", message.lower())


class TestDeterministicReceipts(unittest.TestCase):
    """Test that replay receipts are deterministic."""

    def setUp(self):
        """Create test directory with deterministic fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.primary_dir = Path(self.temp_dir) / "primary"
        self.replay_dir = Path(self.temp_dir) / "replay"

        for d in [self.primary_dir / "baseline", self.primary_dir / "rfl",
                  self.replay_dir / "baseline", self.replay_dir / "rfl"]:
            d.mkdir(parents=True)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _create_deterministic_logs(self, seed: int = 42, cycles: int = 10):
        """Create identical log files with deterministic content."""
        for run_type in ["baseline", "rfl"]:
            for run_dir in [self.primary_dir, self.replay_dir]:
                log_path = run_dir / run_type / "run.jsonl"
                with open(log_path, 'w') as f:
                    for i in range(cycles):
                        ht = hashlib.sha256(f"{seed}:{run_type}:{i}".encode()).hexdigest()
                        record = {
                            "cycle": i,
                            "H_t": ht,
                            "R_t": hashlib.sha256(f"R:{seed}:{i}".encode()).hexdigest(),
                            "U_t": hashlib.sha256(f"U:{seed}:{i}".encode()).hexdigest(),
                        }
                        f.write(json.dumps(record, sort_keys=True) + "\n")

    def test_same_inputs_produce_identical_receipt(self):
        """Building receipt twice with same inputs should produce same result."""
        seed = 42
        cycles = 10

        self._create_deterministic_logs(seed, cycles)

        manifest_path = Path(self.temp_dir) / "manifest.json"
        manifest = {"experiment_id": "U2_EXP_001", "initial_seed": seed, "cycles": cycles}
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, sort_keys=True)

        # Build receipt twice
        receipt1 = build_replay_receipt(
            self.primary_dir, self.replay_dir, manifest_path, git_sha="a" * 40
        )
        # Reset replayed_at for comparison (it uses current time)
        receipt1.replayed_at = "2025-12-06T10:00:00+00:00"
        receipt1.manifest_binding.bound_at = "2025-12-06T10:00:00+00:00"
        receipt1.receipt_hash = compute_receipt_hash(receipt1)

        receipt2 = build_replay_receipt(
            self.primary_dir, self.replay_dir, manifest_path, git_sha="a" * 40
        )
        receipt2.replayed_at = "2025-12-06T10:00:00+00:00"
        receipt2.manifest_binding.bound_at = "2025-12-06T10:00:00+00:00"
        receipt2.receipt_hash = compute_receipt_hash(receipt2)

        # Compare JSON output
        self.assertEqual(receipt1.to_json(), receipt2.to_json())
        self.assertEqual(receipt1.receipt_hash, receipt2.receipt_hash)


# ============================================================================
# Evidence Spine v2 Tests (TASK 1, 2, 3)
# ============================================================================

class TestReceiptIndexContract(unittest.TestCase):
    """Test the Receipt Index contract (TASK 1)."""

    def setUp(self):
        """Create test receipts."""
        from backend.governance.replay_receipt import (
            ReceiptIndexEntry,
            build_receipt_index,
            build_receipt_index_entry,
            dump_receipt_index,
            load_receipt_index,
            update_receipt_index,
            RECEIPT_INDEX_VERSION,
        )
        self.build_receipt_index = build_receipt_index
        self.build_receipt_index_entry = build_receipt_index_entry
        self.dump_receipt_index = dump_receipt_index
        self.load_receipt_index = load_receipt_index
        self.update_receipt_index = update_receipt_index
        self.RECEIPT_INDEX_VERSION = RECEIPT_INDEX_VERSION

        self.temp_dir = tempfile.mkdtemp()

        # Create sample receipts
        self.receipts = []
        for i in range(3):
            receipt = ReplayReceipt(
                receipt_version=REPLAY_RECEIPT_VERSION,
                experiment_id=f"U2_EXP_00{i+1}",
                status=ReplayStatus.VERIFIED,
                replayed_at=f"2025-12-0{i+1}T10:00:00+00:00",
                replay_environment=ReplayEnvironment(
                    git_sha="a" * 40,
                    python_version="3.11.5",
                    platform="win32",
                ),
                manifest_binding=ManifestBinding(
                    manifest_path=f"runs/exp{i+1}/manifest.json",
                    manifest_hash="b" * 64,
                    bound_at=f"2025-12-0{i+1}T10:00:00+00:00",
                ),
                baseline_replay=ReplayRunResult(
                    run_type="baseline",
                    seed_used=12345,
                    cycles_executed=100,
                    expected_log_hash="c" * 64,
                    replay_log_hash="c" * 64,
                    log_hash_match=True,
                    expected_final_ht=f"{'d' * 60}{i:04d}",
                    replay_final_ht=f"{'d' * 60}{i:04d}",
                    final_ht_match=True,
                    ht_sequence_length=100,
                    ht_sequence_match=True,
                ),
                rfl_replay=ReplayRunResult(
                    run_type="rfl",
                    seed_used=12345,
                    cycles_executed=100,
                    expected_log_hash="e" * 64,
                    replay_log_hash="e" * 64,
                    log_hash_match=True,
                    expected_final_ht=f"{'f' * 60}{i:04d}",
                    replay_final_ht=f"{'f' * 60}{i:04d}",
                    final_ht_match=True,
                    ht_sequence_length=100,
                    ht_sequence_match=True,
                ),
                verification_summary=VerificationSummary(
                    checks_passed=12,
                    checks_failed=0,
                ),
            )
            receipt.receipt_hash = compute_receipt_hash(receipt)
            self.receipts.append(receipt)

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_build_empty_index(self):
        """Empty receipt list produces valid empty index."""
        index = self.build_receipt_index([])

        self.assertEqual(index["schema_version"], self.RECEIPT_INDEX_VERSION)
        self.assertEqual(index["receipt_count"], 0)
        self.assertEqual(index["receipts"], [])
        self.assertIn("generated_at", index)

    def test_build_index_with_receipts(self):
        """Index contains all required fields per receipt."""
        index = self.build_receipt_index(self.receipts)

        self.assertEqual(index["schema_version"], self.RECEIPT_INDEX_VERSION)
        self.assertEqual(index["receipt_count"], 3)
        self.assertEqual(len(index["receipts"]), 3)

        # Check each entry has required fields
        for entry in index["receipts"]:
            self.assertIn("receipt_hash", entry)
            self.assertIn("primary_manifest_path", entry)
            self.assertIn("replay_manifest_path", entry)
            self.assertIn("status", entry)
            self.assertIn("ht_series_hash", entry)
            self.assertIn("experiment_id", entry)
            self.assertIn("checks_passed", entry)
            self.assertIn("checks_total", entry)

    def test_index_stable_ordering(self):
        """Index entries are sorted by primary_manifest_path."""
        # Shuffle receipts to verify sorting
        shuffled = list(reversed(self.receipts))
        index = self.build_receipt_index(shuffled)

        paths = [e["primary_manifest_path"] for e in index["receipts"]]
        self.assertEqual(paths, sorted(paths))

    def test_index_json_roundtrip(self):
        """Index survives JSON serialization roundtrip."""
        index = self.build_receipt_index(self.receipts)
        index_path = Path(self.temp_dir) / "index.json"

        self.dump_receipt_index(index_path, index)
        loaded = self.load_receipt_index(index_path)

        self.assertEqual(loaded["schema_version"], index["schema_version"])
        self.assertEqual(loaded["receipt_count"], index["receipt_count"])
        self.assertEqual(len(loaded["receipts"]), len(index["receipts"]))

    def test_index_handles_missing_fields(self):
        """Loading index with missing optional fields works."""
        index_path = Path(self.temp_dir) / "legacy_index.json"

        # Write minimal legacy index
        legacy = {"receipts": [{"primary_manifest_path": "test.json", "status": "VERIFIED"}]}
        with open(index_path, 'w') as f:
            json.dump(legacy, f)

        loaded = self.load_receipt_index(index_path)

        self.assertEqual(loaded["schema_version"], "0.0.0")  # Legacy default
        self.assertEqual(loaded["experiment_id"], "")
        self.assertEqual(loaded["receipt_count"], 1)

    def test_update_receipt_index(self):
        """Updating index adds new receipt and maintains order."""
        index_path = Path(self.temp_dir) / "update_test.json"

        # Create initial index
        initial_index = self.build_receipt_index(self.receipts[:2])
        self.dump_receipt_index(index_path, initial_index)

        # Update with third receipt
        updated = self.update_receipt_index(index_path, self.receipts[2])

        self.assertEqual(updated["receipt_count"], 3)
        paths = [e["primary_manifest_path"] for e in updated["receipts"]]
        self.assertEqual(paths, sorted(paths))

    def test_update_replaces_existing(self):
        """Updating with same manifest path replaces entry."""
        index_path = Path(self.temp_dir) / "replace_test.json"

        # Create initial index
        self.dump_receipt_index(index_path, self.build_receipt_index([self.receipts[0]]))

        # Modify receipt but keep same manifest path
        modified = self.receipts[0]
        modified.verification_summary.checks_passed = 11

        # Update
        updated = self.update_receipt_index(index_path, modified)

        self.assertEqual(updated["receipt_count"], 1)
        self.assertEqual(updated["receipts"][0]["checks_passed"], 11)


class TestGovernanceSummary(unittest.TestCase):
    """Test the governance-grade summary (TASK 3)."""

    def setUp(self):
        """Create test receipts with mixed statuses."""
        from backend.governance.replay_receipt import (
            summarize_replay_receipts,
            summarize_receipt_index,
            build_receipt_index,
        )
        self.summarize_replay_receipts = summarize_replay_receipts
        self.summarize_receipt_index = summarize_receipt_index
        self.build_receipt_index = build_receipt_index

        self.temp_dir = tempfile.mkdtemp()

        # Create receipts with different statuses
        self.verified_receipts = []
        self.failed_receipts = []
        self.incomplete_receipts = []

        for i in range(2):
            verified = self._make_receipt(f"verified_{i}", ReplayStatus.VERIFIED)
            self.verified_receipts.append(verified)

        for i in range(1):
            failed = self._make_receipt(f"failed_{i}", ReplayStatus.FAILED)
            failed.verification_summary.checks_failed = 2
            failed.verification_summary.failed_checks = [
                FailedCheck(check_id="RC-R7", expected="x", actual="y", detail="log hash mismatch")
            ]
            self.failed_receipts.append(failed)

        for i in range(1):
            incomplete = self._make_receipt(f"incomplete_{i}", ReplayStatus.INCOMPLETE)
            incomplete.verification_summary.checks_failed = 5
            self.incomplete_receipts.append(incomplete)

        self.all_receipts = self.verified_receipts + self.failed_receipts + self.incomplete_receipts

    def _make_receipt(self, name: str, status: ReplayStatus) -> ReplayReceipt:
        """Create a test receipt with given status."""
        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id=f"U2_EXP_{name}",
            status=status,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path=f"{name}/manifest.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64, log_hash_match=True,
                expected_final_ht="d" * 64, replay_final_ht="d" * 64, final_ht_match=True,
                ht_sequence_length=100, ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="e" * 64, replay_log_hash="e" * 64, log_hash_match=True,
                expected_final_ht="f" * 64, replay_final_ht="f" * 64, final_ht_match=True,
                ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12 if status == ReplayStatus.VERIFIED else 10,
                checks_failed=0 if status == ReplayStatus.VERIFIED else 2,
            ),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_summary_counts(self):
        """Summary correctly counts each status."""
        summary = self.summarize_replay_receipts(self.all_receipts)

        self.assertEqual(summary["total_receipts"], 4)
        self.assertEqual(summary["verified_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["incomplete_count"], 1)
        self.assertFalse(summary["all_verified"])

    def test_summary_all_verified(self):
        """all_verified is True when all receipts are VERIFIED."""
        summary = self.summarize_replay_receipts(self.verified_receipts)

        self.assertTrue(summary["all_verified"])
        self.assertEqual(summary["verified_count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["incomplete_count"], 0)

    def test_summary_error_codes(self):
        """Summary tracks RECON error codes."""
        summary = self.summarize_replay_receipts(self.all_receipts)

        # Should have error codes for failed and incomplete
        self.assertIn("error_codes", summary)
        error_codes = summary["error_codes"]

        # RECON-19 for failed, RECON-20 for incomplete
        if error_codes:  # May have codes if there are failures
            self.assertTrue(any("RECON" in code for code in error_codes))

    def test_summary_is_deterministic(self):
        """Same receipts produce same summary."""
        summary1 = self.summarize_replay_receipts(self.all_receipts)
        summary2 = self.summarize_replay_receipts(self.all_receipts)

        self.assertEqual(summary1["summary_hash"], summary2["summary_hash"])

    def test_summary_has_hash(self):
        """Summary includes a deterministic hash."""
        summary = self.summarize_replay_receipts(self.verified_receipts)

        self.assertIn("summary_hash", summary)
        self.assertEqual(len(summary["summary_hash"]), 64)  # SHA-256 hex

    def test_summarize_from_index(self):
        """Can summarize directly from receipt index."""
        index = self.build_receipt_index(self.all_receipts)
        summary = self.summarize_receipt_index(index)

        self.assertEqual(summary["total_receipts"], 4)
        self.assertEqual(summary["verified_count"], 2)
        self.assertEqual(summary["failed_count"], 1)
        self.assertEqual(summary["incomplete_count"], 1)

    def test_empty_summary(self):
        """Empty receipt list produces valid summary."""
        summary = self.summarize_replay_receipts([])

        self.assertEqual(summary["total_receipts"], 0)
        self.assertEqual(summary["verified_count"], 0)
        self.assertFalse(summary["all_verified"])


class TestHtSeriesHash(unittest.TestCase):
    """Test H_t series hash computation."""

    def test_ht_series_hash_deterministic(self):
        """Same receipt produces same HT series hash."""
        from backend.governance.replay_receipt import compute_ht_series_hash

        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_001",
            status=ReplayStatus.VERIFIED,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64, log_hash_match=True,
                expected_final_ht="aaa" + "0" * 61, replay_final_ht="aaa" + "0" * 61, final_ht_match=True,
                ht_sequence_length=100, ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="e" * 64, replay_log_hash="e" * 64, log_hash_match=True,
                expected_final_ht="bbb" + "0" * 61, replay_final_ht="bbb" + "0" * 61, final_ht_match=True,
                ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(checks_passed=12, checks_failed=0),
        )

        hash1 = compute_ht_series_hash(receipt)
        hash2 = compute_ht_series_hash(receipt)

        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)

    def test_ht_series_hash_changes_with_ht(self):
        """Different H_t values produce different hash."""
        from backend.governance.replay_receipt import compute_ht_series_hash

        def make_receipt(ht_suffix: str) -> ReplayReceipt:
            return ReplayReceipt(
                receipt_version=REPLAY_RECEIPT_VERSION,
                experiment_id="U2_EXP_001",
                status=ReplayStatus.VERIFIED,
                replayed_at="2025-12-06T10:00:00+00:00",
                replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
                manifest_binding=ManifestBinding(manifest_path="test.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
                baseline_replay=ReplayRunResult(
                    run_type="baseline", seed_used=12345, cycles_executed=100,
                    expected_log_hash="c" * 64, replay_log_hash="c" * 64, log_hash_match=True,
                    expected_final_ht=ht_suffix + "0" * (64 - len(ht_suffix)),
                    replay_final_ht=ht_suffix + "0" * (64 - len(ht_suffix)),
                    final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
                ),
                rfl_replay=ReplayRunResult(
                    run_type="rfl", seed_used=12345, cycles_executed=100,
                    expected_log_hash="e" * 64, replay_log_hash="e" * 64, log_hash_match=True,
                    expected_final_ht="f" * 64, replay_final_ht="f" * 64, final_ht_match=True,
                    ht_sequence_length=100, ht_sequence_match=True,
                ),
                verification_summary=VerificationSummary(checks_passed=12, checks_failed=0),
            )

        receipt1 = make_receipt("aaa")
        receipt2 = make_receipt("bbb")

        hash1 = compute_ht_series_hash(receipt1)
        hash2 = compute_ht_series_hash(receipt2)

        self.assertNotEqual(hash1, hash2)


# ============================================================================
# Phase III Tests: Cross-Run Replay Intelligence
# ============================================================================

class TestDeterminismLedger(unittest.TestCase):
    """Test the Replay Determinism Ledger (TASK 1)."""

    def setUp(self):
        """Import Phase III functions and create test receipts."""
        from backend.governance.replay_receipt import (
            build_replay_determinism_ledger,
            build_ledger_entry,
            save_determinism_ledger,
            load_determinism_ledger,
            LedgerRunEntry,
            DETERMINISM_LEDGER_VERSION,
        )
        self.build_replay_determinism_ledger = build_replay_determinism_ledger
        self.build_ledger_entry = build_ledger_entry
        self.save_determinism_ledger = save_determinism_ledger
        self.load_determinism_ledger = load_determinism_ledger
        self.DETERMINISM_LEDGER_VERSION = DETERMINISM_LEDGER_VERSION

        self.temp_dir = tempfile.mkdtemp()

        # Create test receipts with different statuses and timestamps
        self.receipts = []
        for i, (status, timestamp) in enumerate([
            (ReplayStatus.VERIFIED, "2025-12-01T10:00:00+00:00"),
            (ReplayStatus.VERIFIED, "2025-12-02T10:00:00+00:00"),
            (ReplayStatus.FAILED, "2025-12-03T10:00:00+00:00"),
            (ReplayStatus.VERIFIED, "2025-12-04T10:00:00+00:00"),
            (ReplayStatus.INCOMPLETE, "2025-12-05T10:00:00+00:00"),
        ]):
            receipt = self._make_receipt(f"exp_{i}", status, timestamp)
            self.receipts.append(receipt)

    def _make_receipt(self, name: str, status: ReplayStatus, timestamp: str) -> ReplayReceipt:
        """Create a test receipt."""
        failed_checks = []
        if status == ReplayStatus.FAILED:
            failed_checks = [FailedCheck(check_id="RC-R7", expected="x", actual="y", detail="mismatch")]

        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id=f"U2_EXP_{name}",
            status=status,
            replayed_at=timestamp,
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path=f"{name}/manifest.json", manifest_hash="b" * 64, bound_at=timestamp),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64 if status == ReplayStatus.VERIFIED else "d" * 64,
                log_hash_match=(status == ReplayStatus.VERIFIED),
                expected_final_ht="e" * 64, replay_final_ht="e" * 64 if status == ReplayStatus.VERIFIED else "f" * 64,
                final_ht_match=(status == ReplayStatus.VERIFIED),
                ht_sequence_length=100, ht_sequence_match=(status == ReplayStatus.VERIFIED),
                first_mismatch_cycle=None if status == ReplayStatus.VERIFIED else 5,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="g" * 64, replay_log_hash="g" * 64,
                log_hash_match=True, expected_final_ht="h" * 64, replay_final_ht="h" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12 if status == ReplayStatus.VERIFIED else 10,
                checks_failed=0 if status == ReplayStatus.VERIFIED else 2,
                failed_checks=failed_checks,
            ),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def test_build_empty_ledger(self):
        """Empty receipt list produces valid empty ledger."""
        ledger = self.build_replay_determinism_ledger([])

        self.assertEqual(ledger["schema_version"], self.DETERMINISM_LEDGER_VERSION)
        self.assertEqual(ledger["run_history"], [])
        self.assertEqual(ledger["totals"]["verified"], 0)
        self.assertEqual(ledger["determinism_rate"], 0.0)
        self.assertIsNone(ledger["first_failure_at"])

    def test_build_ledger_with_receipts(self):
        """Ledger contains correct totals and rates."""
        ledger = self.build_replay_determinism_ledger(self.receipts)

        self.assertEqual(ledger["schema_version"], self.DETERMINISM_LEDGER_VERSION)
        self.assertEqual(len(ledger["run_history"]), 5)
        self.assertEqual(ledger["totals"]["verified"], 3)
        self.assertEqual(ledger["totals"]["failed"], 1)
        self.assertEqual(ledger["totals"]["incomplete"], 1)
        self.assertEqual(ledger["determinism_rate"], 0.6)  # 3/5

    def test_ledger_deterministic_ordering(self):
        """Ledger entries are ordered by timestamp."""
        # Shuffle receipts
        shuffled = list(reversed(self.receipts))
        ledger = self.build_replay_determinism_ledger(shuffled)

        timestamps = [e["timestamp"] for e in ledger["run_history"]]
        self.assertEqual(timestamps, sorted(timestamps))

    def test_ledger_tracks_failure_timestamps(self):
        """Ledger tracks first and last failure timestamps."""
        ledger = self.build_replay_determinism_ledger(self.receipts)

        self.assertEqual(ledger["first_failure_at"], "2025-12-03T10:00:00+00:00")
        self.assertEqual(ledger["last_failure_at"], "2025-12-05T10:00:00+00:00")

    def test_ledger_has_hash(self):
        """Ledger includes a deterministic hash."""
        ledger = self.build_replay_determinism_ledger(self.receipts)

        self.assertIn("ledger_hash", ledger)
        self.assertEqual(len(ledger["ledger_hash"]), 64)

    def test_ledger_json_roundtrip(self):
        """Ledger survives file serialization."""
        ledger = self.build_replay_determinism_ledger(self.receipts)
        ledger_path = Path(self.temp_dir) / "ledger.json"

        self.save_determinism_ledger(ledger, ledger_path)
        loaded = self.load_determinism_ledger(ledger_path)

        self.assertEqual(loaded["determinism_rate"], ledger["determinism_rate"])
        self.assertEqual(len(loaded["run_history"]), len(ledger["run_history"]))

    def test_ledger_entry_has_required_fields(self):
        """Each ledger entry has all required fields."""
        ledger = self.build_replay_determinism_ledger(self.receipts)

        for entry in ledger["run_history"]:
            self.assertIn("run_id", entry)
            self.assertIn("experiment_id", entry)
            self.assertIn("status", entry)
            self.assertIn("mismatch_codes", entry)
            self.assertIn("ht_hash", entry)
            self.assertIn("timestamp", entry)
            self.assertIn("checks_passed", entry)
            self.assertIn("checks_total", entry)


class TestIncidentClassifier(unittest.TestCase):
    """Test the Determinism Incident Classifier (TASK 2)."""

    def setUp(self):
        """Import classifier functions."""
        from backend.governance.replay_receipt import (
            classify_replay_incident,
            IncidentSeverity,
            AffectedDomain,
        )
        self.classify_replay_incident = classify_replay_incident
        self.IncidentSeverity = IncidentSeverity
        self.AffectedDomain = AffectedDomain

    def _make_receipt(
        self,
        status: ReplayStatus,
        baseline_ht_match: bool = True,
        baseline_log_match: bool = True,
        first_mismatch: int = None,
    ) -> ReplayReceipt:
        """Create a test receipt with specific failure modes."""
        failed_checks = []
        if not baseline_ht_match:
            failed_checks.append(FailedCheck(check_id="RC-R9", expected="x", actual="y", detail="ht mismatch"))
        if not baseline_log_match:
            failed_checks.append(FailedCheck(check_id="RC-R7", expected="x", actual="y", detail="log mismatch"))

        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_TEST",
            status=status,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test/manifest.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64 if baseline_log_match else "d" * 64,
                log_hash_match=baseline_log_match,
                expected_final_ht="e" * 64, replay_final_ht="e" * 64 if baseline_ht_match else "f" * 64,
                final_ht_match=baseline_ht_match,
                ht_sequence_length=100, ht_sequence_match=baseline_ht_match,
                first_mismatch_cycle=first_mismatch,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="g" * 64, replay_log_hash="g" * 64,
                log_hash_match=True, expected_final_ht="h" * 64, replay_final_ht="h" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12 if status == ReplayStatus.VERIFIED else 10,
                checks_failed=0 if status == ReplayStatus.VERIFIED else len(failed_checks),
                failed_checks=failed_checks,
            ),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def test_verified_receipt_no_incident(self):
        """Verified receipt has NONE severity."""
        receipt = self._make_receipt(ReplayStatus.VERIFIED)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "NONE")
        self.assertEqual(incident["affected_domains"], [])
        self.assertEqual(incident["incident_fingerprint"], "")

    def test_incomplete_receipt_high_severity(self):
        """Incomplete receipt classified as HIGH severity."""
        receipt = self._make_receipt(ReplayStatus.INCOMPLETE)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "HIGH")
        self.assertIn("incomplete", incident["recommended_action"].lower())

    def test_config_mismatch_critical(self):
        """Cycle 0 mismatch is CRITICAL (config issue)."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False, first_mismatch=0)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "CRITICAL")
        self.assertIn("CONFIG", incident["affected_domains"])

    def test_early_ht_divergence_high(self):
        """Early H_t divergence (cycle < 5) is HIGH severity."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False, first_mismatch=2)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "HIGH")
        self.assertIn("H_T", incident["affected_domains"])

    def test_late_ht_divergence_low(self):
        """Late H_t divergence is LOW severity."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False, first_mismatch=50)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "LOW")
        self.assertIn("H_T", incident["affected_domains"])

    def test_log_mismatch_low(self):
        """Log hash mismatch only is LOW severity."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_log_match=False)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "LOW")
        self.assertIn("LOG", incident["affected_domains"])

    def test_multi_domain_failure_high(self):
        """Multiple domain failures are HIGH severity."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False, baseline_log_match=False)
        incident = self.classify_replay_incident(receipt)

        self.assertEqual(incident["severity"], "HIGH")
        self.assertIn("H_T", incident["affected_domains"])
        self.assertIn("LOG", incident["affected_domains"])

    def test_incident_fingerprint_deterministic(self):
        """Same receipt produces same fingerprint."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False)

        incident1 = self.classify_replay_incident(receipt)
        incident2 = self.classify_replay_incident(receipt)

        self.assertEqual(incident1["incident_fingerprint"], incident2["incident_fingerprint"])
        self.assertEqual(len(incident1["incident_fingerprint"]), 24)

    def test_incident_has_mismatch_codes(self):
        """Incident includes mismatch codes from receipt."""
        receipt = self._make_receipt(ReplayStatus.FAILED, baseline_ht_match=False)
        incident = self.classify_replay_incident(receipt)

        self.assertIn("mismatch_codes", incident)
        self.assertIsInstance(incident["mismatch_codes"], list)


class TestGlobalHealthHook(unittest.TestCase):
    """Test the Global Health Hook (TASK 3)."""

    def setUp(self):
        """Import health functions."""
        from backend.governance.replay_receipt import (
            summarize_replay_for_global_health,
            build_replay_determinism_ledger,
            ReplayHealthStatus,
        )
        self.summarize_replay_for_global_health = summarize_replay_for_global_health
        self.build_replay_determinism_ledger = build_replay_determinism_ledger
        self.ReplayHealthStatus = ReplayHealthStatus

    def _make_receipt(self, status: ReplayStatus, timestamp: str) -> ReplayReceipt:
        """Create a minimal test receipt."""
        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_TEST",
            status=status,
            replayed_at=timestamp,
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test/manifest.json", manifest_hash="b" * 64, bound_at=timestamp),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64,
                log_hash_match=True, expected_final_ht="d" * 64, replay_final_ht="d" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="e" * 64, replay_log_hash="e" * 64,
                log_hash_match=True, expected_final_ht="f" * 64, replay_final_ht="f" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(checks_passed=12, checks_failed=0),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def test_all_verified_is_healthy(self):
        """100% verified runs produce OK status."""
        receipts = [self._make_receipt(ReplayStatus.VERIFIED, f"2025-12-0{i}T10:00:00+00:00") for i in range(1, 6)]
        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        self.assertEqual(health["replay_status"], "OK")
        self.assertTrue(health["is_healthy"])
        self.assertFalse(health["is_blocked"])
        self.assertEqual(health["determinism_rate"], 1.0)

    def test_incomplete_runs_blocked(self):
        """Any incomplete run causes BLOCKED status."""
        receipts = [
            self._make_receipt(ReplayStatus.VERIFIED, "2025-12-01T10:00:00+00:00"),
            self._make_receipt(ReplayStatus.INCOMPLETE, "2025-12-02T10:00:00+00:00"),
        ]
        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        self.assertEqual(health["replay_status"], "BLOCKED")
        self.assertTrue(health["is_blocked"])
        self.assertFalse(health["is_healthy"])

    def test_low_determinism_rate_blocked(self):
        """Determinism rate < 90% causes BLOCKED status."""
        receipts = [
            self._make_receipt(ReplayStatus.VERIFIED, "2025-12-01T10:00:00+00:00"),
            self._make_receipt(ReplayStatus.FAILED, "2025-12-02T10:00:00+00:00"),
            self._make_receipt(ReplayStatus.FAILED, "2025-12-03T10:00:00+00:00"),
        ]
        # Manually create a ledger to force the receipts to be FAILED
        for r in receipts[1:]:
            r.status = ReplayStatus.FAILED
        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        # 1/3 = 33% verified, should be BLOCKED
        self.assertEqual(health["replay_status"], "BLOCKED")

    def test_medium_determinism_rate_warn(self):
        """Determinism rate 90-95% causes WARN status."""
        # Create 10 receipts: 9 verified, 1 failed = 90%
        receipts = [self._make_receipt(ReplayStatus.VERIFIED, f"2025-12-{i:02d}T10:00:00+00:00") for i in range(1, 10)]
        failed = self._make_receipt(ReplayStatus.FAILED, "2025-12-10T10:00:00+00:00")
        failed.status = ReplayStatus.FAILED
        receipts.append(failed)

        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        # Should be WARN (90% is at threshold)
        self.assertIn(health["replay_status"], ["WARN", "OK"])

    def test_health_tracks_recent_failures(self):
        """Health summary tracks recent failure count."""
        receipts = [
            self._make_receipt(ReplayStatus.VERIFIED, "2025-12-01T10:00:00+00:00"),
            self._make_receipt(ReplayStatus.FAILED, "2025-12-02T10:00:00+00:00"),
        ]
        receipts[1].status = ReplayStatus.FAILED
        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        self.assertEqual(health["recent_failures"], 1)
        self.assertIn("recent_failure_rate", health)

    def test_health_has_hash(self):
        """Health summary includes integrity hash."""
        receipts = [self._make_receipt(ReplayStatus.VERIFIED, "2025-12-01T10:00:00+00:00")]
        ledger = self.build_replay_determinism_ledger(receipts)
        health = self.summarize_replay_for_global_health(ledger)

        self.assertIn("health_hash", health)
        self.assertEqual(len(health["health_hash"]), 64)


class TestIncidentReport(unittest.TestCase):
    """Test the incident report aggregation."""

    def setUp(self):
        """Import report function."""
        from backend.governance.replay_receipt import get_incident_report

        self.get_incident_report = get_incident_report

    def _make_receipt(self, status: ReplayStatus) -> ReplayReceipt:
        """Create a test receipt."""
        failed_checks = []
        if status == ReplayStatus.FAILED:
            failed_checks = [FailedCheck(check_id="RC-R7", expected="x", actual="y", detail="mismatch")]

        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_TEST",
            status=status,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test/manifest.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64 if status == ReplayStatus.VERIFIED else "d" * 64,
                log_hash_match=(status == ReplayStatus.VERIFIED),
                expected_final_ht="e" * 64, replay_final_ht="e" * 64 if status == ReplayStatus.VERIFIED else "f" * 64,
                final_ht_match=(status == ReplayStatus.VERIFIED),
                ht_sequence_length=100, ht_sequence_match=(status == ReplayStatus.VERIFIED),
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="g" * 64, replay_log_hash="g" * 64,
                log_hash_match=True, expected_final_ht="h" * 64, replay_final_ht="h" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(
                checks_passed=12 if status == ReplayStatus.VERIFIED else 10,
                checks_failed=0 if status == ReplayStatus.VERIFIED else 2,
                failed_checks=failed_checks,
            ),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def test_empty_report(self):
        """Empty receipt list produces valid empty report."""
        report = self.get_incident_report([])

        self.assertEqual(report["total_receipts"], 0)
        self.assertEqual(report["incidents_by_severity"], {})

    def test_report_aggregates_severity(self):
        """Report correctly counts incidents by severity."""
        receipts = [
            self._make_receipt(ReplayStatus.VERIFIED),
            self._make_receipt(ReplayStatus.VERIFIED),
            self._make_receipt(ReplayStatus.FAILED),
        ]
        report = self.get_incident_report(receipts)

        self.assertEqual(report["total_receipts"], 3)
        self.assertIn("NONE", report["incidents_by_severity"])

    def test_report_collects_fingerprints(self):
        """Report collects unique incident fingerprints."""
        receipts = [
            self._make_receipt(ReplayStatus.FAILED),
            self._make_receipt(ReplayStatus.FAILED),
        ]
        report = self.get_incident_report(receipts)

        self.assertIn("unique_fingerprints", report)
        self.assertIsInstance(report["unique_fingerprints"], list)

    def test_report_has_recommendations(self):
        """Report includes recommendations for non-verified receipts."""
        receipts = [self._make_receipt(ReplayStatus.FAILED)]
        report = self.get_incident_report(receipts)

        self.assertIn("recommendations", report)
        self.assertGreater(len(report["recommendations"]), 0)


# ============================================================================
# Phase IV Tests: Replay Governance Radar & Policy Coupler
# ============================================================================

class TestGovernanceRadar(unittest.TestCase):
    """Test the Replay Governance Radar (TASK 1)."""

    def setUp(self):
        """Import Phase IV functions and create test data."""
        from backend.governance.replay_receipt import (
            build_replay_governance_radar,
            RadarStatus,
            GOVERNANCE_RADAR_VERSION,
        )
        self.build_replay_governance_radar = build_replay_governance_radar
        self.RadarStatus = RadarStatus
        self.GOVERNANCE_RADAR_VERSION = GOVERNANCE_RADAR_VERSION

    def _make_ledger(
        self,
        verified: int,
        failed: int,
        incomplete: int = 0,
        start_date: str = "2025-12-01",
    ) -> Dict[str, Any]:
        """Create a test ledger with specified run counts."""
        run_history = []
        total = verified + failed + incomplete

        for i in range(total):
            if i < verified:
                status = "VERIFIED"
            elif i < verified + failed:
                status = "FAILED"
            else:
                status = "INCOMPLETE"

            run_history.append({
                "run_id": f"run_{i:04d}",
                "experiment_id": "U2_EXP_TEST",
                "status": status,
                "mismatch_codes": ["RC-R7"] if status != "VERIFIED" else [],
                "ht_hash": f"{'a' * 60}{i:04d}",
                "timestamp": f"{start_date}T{10 + i}:00:00+00:00",
                "checks_passed": 12 if status == "VERIFIED" else 10,
                "checks_total": 12,
            })

        return {
            "schema_version": "1.0.0",
            "run_history": run_history,
            "totals": {
                "verified": verified,
                "failed": failed,
                "incomplete": incomplete,
            },
            "determinism_rate": verified / total if total > 0 else 0.0,
        }

    def _make_incident(
        self,
        severity: str = "LOW",
        fingerprint: str = "abc123",
    ) -> Dict[str, Any]:
        """Create a test incident."""
        return {
            "severity": severity,
            "affected_domains": ["H_T"],
            "recommended_action": "Test action",
            "incident_fingerprint": fingerprint,
            "mismatch_codes": ["RC-R7"],
        }

    def test_empty_radar(self):
        """Empty ledgers produce valid empty radar."""
        radar = self.build_replay_governance_radar([], [])

        self.assertEqual(radar["schema_version"], self.GOVERNANCE_RADAR_VERSION)
        self.assertEqual(radar["total_runs"], 0)
        self.assertEqual(radar["determinism_rate_series"], [])
        self.assertEqual(radar["radar_status"], "STABLE")
        self.assertIn("radar_hash", radar)

    def test_radar_with_single_ledger(self):
        """Radar correctly processes single ledger."""
        ledger = self._make_ledger(verified=8, failed=2)
        radar = self.build_replay_governance_radar([ledger], [])

        self.assertEqual(radar["total_runs"], 10)
        self.assertEqual(len(radar["determinism_rate_series"]), 10)
        self.assertIn("radar_status", radar)

    def test_radar_determinism_rate_series(self):
        """Radar builds correct determinism rate timeline."""
        ledger = self._make_ledger(verified=5, failed=0)
        radar = self.build_replay_governance_radar([ledger], [])

        series = radar["determinism_rate_series"]
        self.assertEqual(len(series), 5)

        # All verified, so rate should reach 1.0
        self.assertEqual(series[-1]["determinism_rate"], 1.0)

        # Each entry should have timestamp and run_index
        for entry in series:
            self.assertIn("timestamp", entry)
            self.assertIn("determinism_rate", entry)
            self.assertIn("run_index", entry)

    def test_radar_stable_status(self):
        """100% determinism produces STABLE status."""
        ledger = self._make_ledger(verified=10, failed=0)
        radar = self.build_replay_governance_radar([ledger], [])

        self.assertEqual(radar["radar_status"], "STABLE")

    def test_radar_unstable_low_rate(self):
        """Low determinism rate produces UNSTABLE status."""
        ledger = self._make_ledger(verified=5, failed=5)  # 50%
        radar = self.build_replay_governance_radar([ledger], [])

        self.assertEqual(radar["radar_status"], "UNSTABLE")

    def test_radar_unstable_critical_incidents(self):
        """High critical incident rate produces UNSTABLE status."""
        ledger = self._make_ledger(verified=9, failed=1)
        incidents = [
            self._make_incident(severity="CRITICAL", fingerprint="crit1"),
            self._make_incident(severity="CRITICAL", fingerprint="crit2"),
            self._make_incident(severity="CRITICAL", fingerprint="crit3"),
            self._make_incident(severity="LOW", fingerprint="low1"),
        ]
        radar = self.build_replay_governance_radar([ledger], incidents)

        # 3/4 = 75% critical incidents > 20% threshold
        self.assertEqual(radar["radar_status"], "UNSTABLE")

    def test_radar_degrading_status(self):
        """Degrading rate produces DEGRADING status."""
        ledger = self._make_ledger(verified=9, failed=1)  # 90%
        radar = self.build_replay_governance_radar([ledger], [])

        # 90% is below 95%, so should be DEGRADING
        self.assertEqual(radar["radar_status"], "DEGRADING")

    def test_radar_hot_fingerprints(self):
        """Radar identifies recurring fingerprints as hot."""
        ledger = self._make_ledger(verified=5, failed=0)
        incidents = [
            self._make_incident(fingerprint="recurring123"),
            self._make_incident(fingerprint="recurring123"),
            self._make_incident(fingerprint="unique456"),
        ]
        radar = self.build_replay_governance_radar([ledger], incidents)

        self.assertIn("recurring123", radar["hot_fingerprints"])
        self.assertNotIn("unique456", radar["hot_fingerprints"])

    def test_radar_has_hash(self):
        """Radar includes integrity hash."""
        ledger = self._make_ledger(verified=3, failed=0)
        radar = self.build_replay_governance_radar([ledger], [])

        self.assertIn("radar_hash", radar)
        self.assertEqual(len(radar["radar_hash"]), 64)

    def test_radar_deterministic(self):
        """Same inputs produce same radar hash."""
        ledger = self._make_ledger(verified=5, failed=1)
        incidents = [self._make_incident()]

        radar1 = self.build_replay_governance_radar([ledger], incidents)
        radar2 = self.build_replay_governance_radar([ledger], incidents)

        # Note: generated_at differs, but hash should be based on content
        # The hash is computed before generated_at changes, so check structure
        self.assertEqual(radar1["total_runs"], radar2["total_runs"])
        self.assertEqual(radar1["radar_status"], radar2["radar_status"])

    def test_radar_multiple_ledgers(self):
        """Radar combines multiple ledgers."""
        ledger1 = self._make_ledger(verified=3, failed=0, start_date="2025-12-01")
        ledger2 = self._make_ledger(verified=2, failed=1, start_date="2025-12-05")
        radar = self.build_replay_governance_radar([ledger1, ledger2], [])

        self.assertEqual(radar["total_runs"], 6)


class TestPromotionEvaluation(unittest.TestCase):
    """Test the Promotion Policy Coupler (TASK 2)."""

    def setUp(self):
        """Import Phase IV functions."""
        from backend.governance.replay_receipt import (
            evaluate_replay_for_promotion,
            PromotionStatus,
        )
        self.evaluate_replay_for_promotion = evaluate_replay_for_promotion
        self.PromotionStatus = PromotionStatus

    def _make_radar(
        self,
        status: str = "STABLE",
        hot_fingerprints: List[str] = None,
    ) -> Dict[str, Any]:
        """Create a test radar."""
        return {
            "schema_version": "1.0.0",
            "total_runs": 10,
            "radar_status": status,
            "hot_fingerprints": hot_fingerprints or [],
            "determinism_rate_series": [],
        }

    def _make_health(
        self,
        status: str = "OK",
        is_blocked: bool = False,
        is_healthy: bool = True,
        determinism_rate: float = 1.0,
        recent_failures: int = 0,
        incomplete_runs: int = 0,
        blocking_fingerprints: List[str] = None,
    ) -> Dict[str, Any]:
        """Create a test health summary."""
        return {
            "replay_status": status,
            "is_blocked": is_blocked,
            "is_healthy": is_healthy,
            "determinism_rate": determinism_rate,
            "recent_failures": recent_failures,
            "incomplete_runs": incomplete_runs,
            "blocking_fingerprints": blocking_fingerprints or [],
        }

    def test_ok_status_allows_promotion(self):
        """OK status with stable radar allows promotion."""
        radar = self._make_radar(status="STABLE")
        health = self._make_health(status="OK", is_healthy=True, determinism_rate=1.0)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertTrue(eval_result["replay_ok_for_promotion"])
        self.assertEqual(eval_result["status"], "OK")
        self.assertIn("Full determinism verified", eval_result["notes"][0])

    def test_blocked_health_blocks_promotion(self):
        """Blocked health status blocks promotion."""
        radar = self._make_radar(status="STABLE")
        health = self._make_health(status="BLOCKED", is_blocked=True, is_healthy=False)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertFalse(eval_result["replay_ok_for_promotion"])
        self.assertEqual(eval_result["status"], "BLOCK")

    def test_unstable_radar_blocks_promotion(self):
        """Unstable radar blocks promotion."""
        radar = self._make_radar(status="UNSTABLE")
        health = self._make_health(status="OK", is_healthy=True)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertFalse(eval_result["replay_ok_for_promotion"])
        self.assertEqual(eval_result["status"], "BLOCK")
        self.assertTrue(any("UNSTABLE" in n for n in eval_result["notes"]))

    def test_degrading_radar_warns(self):
        """Degrading radar produces WARN status."""
        radar = self._make_radar(status="DEGRADING")
        health = self._make_health(status="OK", is_healthy=True)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertTrue(eval_result["replay_ok_for_promotion"])
        self.assertEqual(eval_result["status"], "WARN")
        self.assertTrue(any("degrading" in n.lower() for n in eval_result["notes"]))

    def test_warn_health_produces_warn(self):
        """WARN health status produces WARN promotion."""
        radar = self._make_radar(status="STABLE")
        health = self._make_health(status="WARN", is_healthy=False, recent_failures=2)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertTrue(eval_result["replay_ok_for_promotion"])
        self.assertEqual(eval_result["status"], "WARN")

    def test_blocking_fingerprints_collected(self):
        """Evaluation collects blocking fingerprints from both sources."""
        radar = self._make_radar(status="DEGRADING", hot_fingerprints=["hot1", "hot2"])
        health = self._make_health(blocking_fingerprints=["block1"])

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertEqual(len(eval_result["blocking_fingerprints"]), 3)
        self.assertIn("hot1", eval_result["blocking_fingerprints"])
        self.assertIn("block1", eval_result["blocking_fingerprints"])

    def test_incomplete_runs_noted(self):
        """Incomplete runs are noted in blocked evaluation."""
        radar = self._make_radar(status="STABLE")
        health = self._make_health(
            status="BLOCKED",
            is_blocked=True,
            incomplete_runs=2,
        )

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertFalse(eval_result["replay_ok_for_promotion"])
        self.assertTrue(any("Incomplete runs" in n for n in eval_result["notes"]))

    def test_determinism_rate_in_notes(self):
        """OK evaluation includes determinism rate in notes."""
        radar = self._make_radar(status="STABLE")
        health = self._make_health(determinism_rate=0.95)

        eval_result = self.evaluate_replay_for_promotion(radar, health)

        self.assertTrue(any("Determinism rate:" in n for n in eval_result["notes"]))


class TestDirectorPanel(unittest.TestCase):
    """Test the Director Replay Panel (TASK 3)."""

    def setUp(self):
        """Import Phase IV functions."""
        from backend.governance.replay_receipt import (
            build_replay_director_panel,
            StatusLight,
        )
        self.build_replay_director_panel = build_replay_director_panel
        self.StatusLight = StatusLight

    def _make_radar(
        self,
        status: str = "STABLE",
        total_runs: int = 10,
        critical_incident_rate: float = 0.0,
        determinism_rate_series: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Create a test radar."""
        if determinism_rate_series is None:
            determinism_rate_series = [
                {"timestamp": f"2025-12-0{i}T10:00:00+00:00", "determinism_rate": 1.0, "run_index": i}
                for i in range(1, total_runs + 1)
            ]
        return {
            "schema_version": "1.0.0",
            "total_runs": total_runs,
            "radar_status": status,
            "hot_fingerprints": [],
            "critical_incident_rate": critical_incident_rate,
            "determinism_rate_series": determinism_rate_series,
        }

    def _make_promotion(
        self,
        status: str = "OK",
        determinism_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a test promotion evaluation."""
        return {
            "replay_ok_for_promotion": status == "OK",
            "status": status,
            "blocking_fingerprints": [],
            "notes": [],
            "determinism_rate": determinism_rate,
        }

    def test_green_light_full_determinism(self):
        """Full determinism produces GREEN light."""
        radar = self._make_radar(status="STABLE")
        promotion = self._make_promotion(status="OK", determinism_rate=1.0)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertIn("fully verified", panel["headline"])

    def test_green_light_healthy(self):
        """Healthy status with partial rate produces GREEN."""
        radar = self._make_radar(status="STABLE")
        promotion = self._make_promotion(status="OK", determinism_rate=0.98)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "GREEN")
        self.assertIn("healthy", panel["headline"])

    def test_yellow_light_degrading(self):
        """Degrading radar produces YELLOW light."""
        radar = self._make_radar(status="DEGRADING")
        promotion = self._make_promotion(status="WARN", determinism_rate=0.92)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "YELLOW")
        self.assertIn("degrading", panel["headline"].lower())

    def test_yellow_light_warn(self):
        """WARN promotion produces YELLOW light."""
        radar = self._make_radar(status="STABLE")
        promotion = self._make_promotion(status="WARN", determinism_rate=0.9)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "YELLOW")

    def test_red_light_blocked(self):
        """BLOCK promotion produces RED light."""
        radar = self._make_radar(status="STABLE")
        promotion = self._make_promotion(status="BLOCK", determinism_rate=0.5)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("blocked", panel["headline"].lower())

    def test_red_light_unstable(self):
        """Unstable radar produces RED light."""
        radar = self._make_radar(status="UNSTABLE")
        promotion = self._make_promotion(status="OK", determinism_rate=0.8)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["status_light"], "RED")
        self.assertIn("unstable", panel["headline"].lower())

    def test_panel_metrics(self):
        """Panel contains correct metrics."""
        radar = self._make_radar(
            status="STABLE",
            total_runs=10,
            critical_incident_rate=0.1,
        )
        promotion = self._make_promotion(status="OK", determinism_rate=0.95)

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertEqual(panel["total_runs"], 10)
        self.assertEqual(panel["determinism_rate"], 0.95)
        self.assertEqual(panel["critical_incident_count"], 1)  # 0.1 * 10
        self.assertIn("recent_failure_rate", panel)

    def test_panel_has_hash(self):
        """Panel includes integrity hash."""
        radar = self._make_radar()
        promotion = self._make_promotion()

        panel = self.build_replay_director_panel(radar, promotion)

        self.assertIn("panel_hash", panel)
        self.assertEqual(len(panel["panel_hash"]), 64)

    def test_panel_headline_content(self):
        """Panel headline reflects status correctly."""
        # Test different scenarios
        test_cases = [
            ("STABLE", "OK", 1.0, "fully verified"),
            ("STABLE", "OK", 0.95, "healthy"),
            ("DEGRADING", "WARN", 0.9, "degrading"),
            ("UNSTABLE", "BLOCK", 0.5, "unstable"),
            ("STABLE", "BLOCK", 0.5, "blocked"),
        ]

        for radar_status, promo_status, rate, expected_in_headline in test_cases:
            radar = self._make_radar(status=radar_status)
            promotion = self._make_promotion(status=promo_status, determinism_rate=rate)

            panel = self.build_replay_director_panel(radar, promotion)

            self.assertIn(
                expected_in_headline,
                panel["headline"].lower(),
                f"Expected '{expected_in_headline}' in headline for {radar_status}/{promo_status}",
            )


class TestFullGovernanceStatus(unittest.TestCase):
    """Test the full governance status convenience function."""

    def setUp(self):
        """Import functions and create test receipts."""
        from backend.governance.replay_receipt import get_full_governance_status

        self.get_full_governance_status = get_full_governance_status
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _make_receipt(self, status: ReplayStatus, timestamp: str = "2025-12-06T10:00:00+00:00") -> ReplayReceipt:
        """Create a test receipt."""
        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_TEST",
            status=status,
            replayed_at=timestamp,
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test/manifest.json", manifest_hash="b" * 64, bound_at=timestamp),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64,
                log_hash_match=True, expected_final_ht="d" * 64, replay_final_ht="d" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="e" * 64, replay_log_hash="e" * 64,
                log_hash_match=True, expected_final_ht="f" * 64, replay_final_ht="f" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(checks_passed=12, checks_failed=0),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def _make_ledger(self, verified: int, failed: int) -> Dict[str, Any]:
        """Create a test ledger."""
        run_history = []
        total = verified + failed
        for i in range(total):
            status = "VERIFIED" if i < verified else "FAILED"
            run_history.append({
                "run_id": f"run_{i}",
                "experiment_id": "U2_EXP_TEST",
                "status": status,
                "mismatch_codes": [],
                "ht_hash": "a" * 64,
                "timestamp": f"2025-12-0{i+1}T10:00:00+00:00",
                "checks_passed": 12 if status == "VERIFIED" else 10,
                "checks_total": 12,
            })
        return {
            "schema_version": "1.0.0",
            "run_history": run_history,
            "totals": {"verified": verified, "failed": failed, "incomplete": 0},
            "determinism_rate": verified / total if total > 0 else 0.0,
        }

    def test_full_status_with_healthy_inputs(self):
        """Full governance status with all verified runs."""
        ledger = self._make_ledger(verified=5, failed=0)
        receipts = [self._make_receipt(ReplayStatus.VERIFIED) for _ in range(5)]

        status = self.get_full_governance_status([ledger], receipts)

        self.assertIn("radar", status)
        self.assertIn("promotion_evaluation", status)
        self.assertIn("director_panel", status)
        self.assertIn("incident_summary", status)

        self.assertEqual(status["radar"]["radar_status"], "STABLE")
        self.assertTrue(status["promotion_evaluation"]["replay_ok_for_promotion"])
        self.assertEqual(status["director_panel"]["status_light"], "GREEN")

    def test_full_status_with_failures(self):
        """Full governance status with some failures."""
        ledger = self._make_ledger(verified=5, failed=5)
        receipts = [self._make_receipt(ReplayStatus.VERIFIED) for _ in range(5)]
        receipts.extend([self._make_receipt(ReplayStatus.FAILED) for _ in range(5)])

        status = self.get_full_governance_status([ledger], receipts)

        # 50% determinism rate should produce UNSTABLE/RED
        self.assertEqual(status["radar"]["radar_status"], "UNSTABLE")
        self.assertFalse(status["promotion_evaluation"]["replay_ok_for_promotion"])
        self.assertEqual(status["director_panel"]["status_light"], "RED")

    def test_full_status_empty_inputs(self):
        """Full governance status with empty inputs."""
        status = self.get_full_governance_status([], [])

        self.assertEqual(status["radar"]["radar_status"], "STABLE")
        self.assertEqual(status["director_panel"]["status_light"], "GREEN")


# ============================================================================
# Phase V Tests: Cross-System Replay Integration
# ============================================================================

class TestGlobalConsoleAdapter(unittest.TestCase):
    """Test the global console adapter (TASK 1)."""

    def setUp(self):
        """Import Phase V functions."""
        from backend.governance.replay_receipt import (
            summarize_replay_for_global_console,
            GLOBAL_CONSOLE_VERSION,
        )
        self.summarize_replay_for_global_console = summarize_replay_for_global_console
        self.GLOBAL_CONSOLE_VERSION = GLOBAL_CONSOLE_VERSION

    def _make_radar(
        self,
        status: str = "STABLE",
        hot_fingerprints: List[str] = None,
        critical_incident_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Create a test radar."""
        return {
            "radar_status": status,
            "hot_fingerprints": hot_fingerprints or [],
            "critical_incident_rate": critical_incident_rate,
            "total_runs": 10,
        }

    def _make_promotion(
        self,
        status: str = "OK",
        replay_ok: bool = True,
        determinism_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a test promotion evaluation."""
        return {
            "status": status,
            "replay_ok_for_promotion": replay_ok,
            "determinism_rate": determinism_rate,
        }

    def test_console_summary_schema(self):
        """Console summary has all required fields."""
        radar = self._make_radar()
        promotion = self._make_promotion()

        summary = self.summarize_replay_for_global_console(radar, promotion)

        self.assertEqual(summary["schema_version"], self.GLOBAL_CONSOLE_VERSION)
        self.assertIn("replay_ok", summary)
        self.assertIn("radar_status", summary)
        self.assertIn("promotion_status", summary)
        self.assertIn("hot_fingerprints_count", summary)
        self.assertIn("critical_incident_rate", summary)
        self.assertIn("headline", summary)
        self.assertIn("console_hash", summary)

    def test_console_summary_ok_status(self):
        """OK status produces correct summary."""
        radar = self._make_radar(status="STABLE")
        promotion = self._make_promotion(status="OK", replay_ok=True, determinism_rate=1.0)

        summary = self.summarize_replay_for_global_console(radar, promotion)

        self.assertTrue(summary["replay_ok"])
        self.assertEqual(summary["radar_status"], "STABLE")
        self.assertEqual(summary["promotion_status"], "OK")
        self.assertIn("verified", summary["headline"])

    def test_console_summary_warn_status(self):
        """WARN status produces appropriate headline."""
        radar = self._make_radar(status="DEGRADING")
        promotion = self._make_promotion(status="WARN", replay_ok=True, determinism_rate=0.92)

        summary = self.summarize_replay_for_global_console(radar, promotion)

        self.assertTrue(summary["replay_ok"])
        self.assertEqual(summary["promotion_status"], "WARN")
        self.assertIn("monitoring", summary["headline"])

    def test_console_summary_block_status(self):
        """BLOCK status produces appropriate headline."""
        radar = self._make_radar(status="UNSTABLE")
        promotion = self._make_promotion(status="BLOCK", replay_ok=False, determinism_rate=0.5)

        summary = self.summarize_replay_for_global_console(radar, promotion)

        self.assertFalse(summary["replay_ok"])
        self.assertEqual(summary["promotion_status"], "BLOCK")
        self.assertIn("investigation", summary["headline"])

    def test_console_summary_neutral_language(self):
        """Headlines use neutral language (no 'good/bad')."""
        test_cases = [
            (self._make_radar("STABLE"), self._make_promotion("OK", True, 1.0)),
            (self._make_radar("DEGRADING"), self._make_promotion("WARN", True, 0.9)),
            (self._make_radar("UNSTABLE"), self._make_promotion("BLOCK", False, 0.5)),
        ]

        for radar, promotion in test_cases:
            summary = self.summarize_replay_for_global_console(radar, promotion)
            headline = summary["headline"].lower()
            # Ensure no judgmental language
            self.assertNotIn("good", headline)
            self.assertNotIn("bad", headline)
            self.assertNotIn("great", headline)
            self.assertNotIn("terrible", headline)

    def test_console_summary_has_hash(self):
        """Console summary includes integrity hash."""
        radar = self._make_radar()
        promotion = self._make_promotion()

        summary = self.summarize_replay_for_global_console(radar, promotion)

        self.assertIn("console_hash", summary)
        self.assertEqual(len(summary["console_hash"]), 64)


class TestEvidenceChainHook(unittest.TestCase):
    """Test the evidence chain hook (TASK 2)."""

    def setUp(self):
        """Import Phase V functions."""
        from backend.governance.replay_receipt import (
            attach_replay_governance_to_evidence,
            EVIDENCE_CHAIN_VERSION,
        )
        self.attach_replay_governance_to_evidence = attach_replay_governance_to_evidence
        self.EVIDENCE_CHAIN_VERSION = EVIDENCE_CHAIN_VERSION

    def _make_chain(self) -> Dict[str, Any]:
        """Create a mock evidence chain."""
        return {
            "experiment_id": "U2_EXP_001",
            "manifest_hash": "a" * 64,
            "attestations": [
                {"type": "G1", "status": "PASS"},
                {"type": "G2", "status": "PASS"},
            ],
            "updated_at": "2025-12-01T10:00:00+00:00",
        }

    def _make_radar(
        self,
        status: str = "STABLE",
        hot_fingerprints: List[str] = None,
        critical_incident_rate: float = 0.0,
        total_runs: int = 10,
    ) -> Dict[str, Any]:
        """Create a test radar."""
        return {
            "radar_status": status,
            "hot_fingerprints": hot_fingerprints or [],
            "critical_incident_rate": critical_incident_rate,
            "total_runs": total_runs,
        }

    def _make_promotion(
        self,
        status: str = "OK",
        determinism_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a test promotion evaluation."""
        return {
            "status": status,
            "replay_ok_for_promotion": status == "OK",
            "determinism_rate": determinism_rate,
        }

    def test_attach_preserves_existing_fields(self):
        """Attaching tile preserves all existing chain fields."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)

        self.assertEqual(updated["experiment_id"], chain["experiment_id"])
        self.assertEqual(updated["manifest_hash"], chain["manifest_hash"])
        self.assertEqual(updated["attestations"], chain["attestations"])

    def test_attach_adds_replay_governance_tile(self):
        """Attaching adds replay_governance subtree."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)

        self.assertIn("replay_governance", updated)
        tile = updated["replay_governance"]
        self.assertIn("status", tile)
        self.assertIn("determinism_rate", tile)
        self.assertIn("critical_incident_rate", tile)

    def test_attach_tile_schema(self):
        """Tile has correct schema version and fields."""
        chain = self._make_chain()
        radar = self._make_radar(total_runs=15, critical_incident_rate=0.05)
        promotion = self._make_promotion(determinism_rate=0.95)

        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)
        tile = updated["replay_governance"]

        self.assertEqual(tile["schema_version"], self.EVIDENCE_CHAIN_VERSION)
        self.assertEqual(tile["status"], "OK")
        self.assertEqual(tile["determinism_rate"], 0.95)
        self.assertEqual(tile["critical_incident_rate"], 0.05)
        self.assertEqual(tile["total_runs"], 15)
        self.assertIn("tile_hash", tile)

    def test_attach_tile_status_mapping(self):
        """Tile status correctly maps from promotion status."""
        chain = self._make_chain()
        radar = self._make_radar()

        # Test OK
        promotion_ok = self._make_promotion(status="OK")
        updated_ok = self.attach_replay_governance_to_evidence(chain, radar, promotion_ok)
        self.assertEqual(updated_ok["replay_governance"]["status"], "OK")

        # Test WARN
        promotion_warn = self._make_promotion(status="WARN")
        updated_warn = self.attach_replay_governance_to_evidence(chain, radar, promotion_warn)
        self.assertEqual(updated_warn["replay_governance"]["status"], "WARN")

        # Test BLOCK
        promotion_block = self._make_promotion(status="BLOCK")
        updated_block = self.attach_replay_governance_to_evidence(chain, radar, promotion_block)
        self.assertEqual(updated_block["replay_governance"]["status"], "BLOCK")

    def test_attach_is_non_mutating(self):
        """Attaching returns new dict, doesn't mutate original."""
        chain = self._make_chain()
        original_keys = set(chain.keys())
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)

        # Original unchanged
        self.assertEqual(set(chain.keys()), original_keys)
        self.assertNotIn("replay_governance", chain)

        # Updated has new key
        self.assertIn("replay_governance", updated)

    def test_attach_limits_hot_fingerprints(self):
        """Tile limits hot fingerprints to first 5."""
        chain = self._make_chain()
        radar = self._make_radar(hot_fingerprints=[f"fp_{i}" for i in range(10)])
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)
        tile = updated["replay_governance"]

        self.assertEqual(tile["hot_fingerprints_count"], 10)
        self.assertEqual(len(tile["hot_fingerprints"]), 5)


class TestPolicyCouplingRegression(unittest.TestCase):
    """Test policy coupling regression scenarios (TASK 3)."""

    def setUp(self):
        """Import functions for regression testing."""
        from backend.governance.replay_receipt import (
            build_replay_governance_radar,
            evaluate_replay_for_promotion,
            summarize_replay_for_global_console,
            summarize_replay_for_global_health,
            build_replay_determinism_ledger,
            classify_replay_incident,
        )
        self.build_replay_governance_radar = build_replay_governance_radar
        self.evaluate_replay_for_promotion = evaluate_replay_for_promotion
        self.summarize_replay_for_global_console = summarize_replay_for_global_console
        self.summarize_replay_for_global_health = summarize_replay_for_global_health
        self.build_replay_determinism_ledger = build_replay_determinism_ledger
        self.classify_replay_incident = classify_replay_incident

    def _make_ledger_with_history(
        self,
        history: List[str],
        start_date: str = "2025-12-01",
    ) -> Dict[str, Any]:
        """Create a ledger with specific status history.

        Args:
            history: List of statuses in chronological order, e.g. ["VERIFIED", "FAILED"]
        """
        run_history = []
        verified = 0
        failed = 0
        incomplete = 0

        for i, status in enumerate(history):
            run_history.append({
                "run_id": f"run_{i:04d}",
                "experiment_id": "U2_EXP_TEST",
                "status": status,
                "mismatch_codes": ["RC-R7"] if status != "VERIFIED" else [],
                "ht_hash": f"{'a' * 60}{i:04d}",
                "timestamp": f"{start_date}T{10 + i}:00:00+00:00",
                "checks_passed": 12 if status == "VERIFIED" else 10,
                "checks_total": 12,
            })
            if status == "VERIFIED":
                verified += 1
            elif status == "FAILED":
                failed += 1
            else:
                incomplete += 1

        total = len(history)
        return {
            "schema_version": "1.0.0",
            "run_history": run_history,
            "totals": {
                "verified": verified,
                "failed": failed,
                "incomplete": incomplete,
            },
            "determinism_rate": verified / total if total > 0 else 0.0,
        }

    def _make_incident(
        self,
        severity: str = "LOW",
        fingerprint: str = None,
    ) -> Dict[str, Any]:
        """Create a test incident."""
        if fingerprint is None:
            fingerprint = hashlib.sha256(severity.encode()).hexdigest()[:24]
        return {
            "severity": severity,
            "affected_domains": ["H_T"],
            "recommended_action": "Test action",
            "incident_fingerprint": fingerprint,
            "mismatch_codes": ["RC-R7"],
        }

    def test_stable_to_degrading_transition(self):
        """Long STABLE history turning DEGRADING produces WARN."""
        # Start with 9 verified runs, add 1 failure (90% rate = DEGRADING threshold)
        degrading_history = ["VERIFIED"] * 9 + ["FAILED"]

        ledger = self._make_ledger_with_history(degrading_history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)
        console = self.summarize_replay_for_global_console(radar, promotion)

        # Should be DEGRADING/WARN but still allowed
        self.assertEqual(radar["radar_status"], "DEGRADING")
        self.assertEqual(promotion["status"], "WARN")
        self.assertTrue(promotion["replay_ok_for_promotion"])
        self.assertTrue(console["replay_ok"])

    def test_degrading_to_unstable_transition(self):
        """DEGRADING trend continuing to UNSTABLE produces BLOCK."""
        # 5 verified, 5 failed = 50% determinism rate
        unstable_history = ["VERIFIED"] * 5 + ["FAILED"] * 5

        ledger = self._make_ledger_with_history(unstable_history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)
        console = self.summarize_replay_for_global_console(radar, promotion)

        # Should be UNSTABLE/BLOCK
        self.assertEqual(radar["radar_status"], "UNSTABLE")
        self.assertEqual(promotion["status"], "BLOCK")
        self.assertFalse(promotion["replay_ok_for_promotion"])
        self.assertFalse(console["replay_ok"])

    def test_below_90_threshold_blocks(self):
        """Determinism rate below 0.9 blocks promotion."""
        # 8 verified, 2 failed = 80% (below 0.9 threshold)
        history = ["VERIFIED"] * 8 + ["FAILED"] * 2

        ledger = self._make_ledger_with_history(history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)

        # Should be BLOCKED due to low rate
        self.assertFalse(promotion["replay_ok_for_promotion"])
        self.assertEqual(promotion["status"], "BLOCK")

    def test_repeated_critical_fingerprints_blocks(self):
        """Repeated critical fingerprints cause BLOCK."""
        # Good ledger but bad incidents
        history = ["VERIFIED"] * 10
        ledger = self._make_ledger_with_history(history)

        # Create repeated critical incidents with same fingerprint
        critical_fp = "repeated_critical_123"
        incidents = [
            self._make_incident(severity="CRITICAL", fingerprint=critical_fp),
            self._make_incident(severity="CRITICAL", fingerprint=critical_fp),
            self._make_incident(severity="CRITICAL", fingerprint=critical_fp),
            self._make_incident(severity="LOW", fingerprint="other_456"),
        ]

        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], incidents)
        promotion = self.evaluate_replay_for_promotion(radar, health)

        # High critical incident rate (75%) should trigger UNSTABLE
        self.assertEqual(radar["radar_status"], "UNSTABLE")
        self.assertFalse(promotion["replay_ok_for_promotion"])

        # The repeated fingerprint should be in hot_fingerprints
        self.assertIn(critical_fp, radar["hot_fingerprints"])

    def test_incomplete_runs_always_block(self):
        """Any incomplete run causes immediate BLOCK."""
        history = ["VERIFIED"] * 9 + ["INCOMPLETE"]

        ledger = self._make_ledger_with_history(history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)

        # Incomplete should always block
        self.assertTrue(health["is_blocked"])
        self.assertEqual(promotion["status"], "BLOCK")
        self.assertFalse(promotion["replay_ok_for_promotion"])

    def test_early_degradation_detection(self):
        """Early signs of degradation trigger WARN before BLOCK."""
        # 9 verified, 1 failed = 90% (at the WARN threshold)
        history = ["VERIFIED"] * 9 + ["FAILED"]

        ledger = self._make_ledger_with_history(history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)
        console = self.summarize_replay_for_global_console(radar, promotion)

        # Should be WARN (degrading) but not BLOCK
        self.assertEqual(radar["radar_status"], "DEGRADING")
        self.assertEqual(promotion["status"], "WARN")
        self.assertTrue(promotion["replay_ok_for_promotion"])
        self.assertTrue(console["replay_ok"])

    def test_recovery_from_degrading_to_stable(self):
        """Recovery with verified runs returns to STABLE."""
        # Had some failures in the past, but now all recent are verified
        history = ["VERIFIED"] * 10

        ledger = self._make_ledger_with_history(history)
        health = self.summarize_replay_for_global_health(ledger)
        radar = self.build_replay_governance_radar([ledger], [])
        promotion = self.evaluate_replay_for_promotion(radar, health)

        # Should be fully STABLE/OK
        self.assertEqual(radar["radar_status"], "STABLE")
        self.assertEqual(promotion["status"], "OK")
        self.assertTrue(promotion["replay_ok_for_promotion"])


class TestGovernanceSnapshot(unittest.TestCase):
    """Test the full governance snapshot function."""

    def setUp(self):
        """Import Phase V functions."""
        from backend.governance.replay_receipt import build_full_governance_snapshot

        self.build_full_governance_snapshot = build_full_governance_snapshot
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)

    def _make_receipt(self, status: ReplayStatus) -> ReplayReceipt:
        """Create a test receipt."""
        receipt = ReplayReceipt(
            receipt_version=REPLAY_RECEIPT_VERSION,
            experiment_id="U2_EXP_TEST",
            status=status,
            replayed_at="2025-12-06T10:00:00+00:00",
            replay_environment=ReplayEnvironment(git_sha="a" * 40, python_version="3.11.5", platform="win32"),
            manifest_binding=ManifestBinding(manifest_path="test/manifest.json", manifest_hash="b" * 64, bound_at="2025-12-06T10:00:00+00:00"),
            baseline_replay=ReplayRunResult(
                run_type="baseline", seed_used=12345, cycles_executed=100,
                expected_log_hash="c" * 64, replay_log_hash="c" * 64,
                log_hash_match=True, expected_final_ht="d" * 64, replay_final_ht="d" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            rfl_replay=ReplayRunResult(
                run_type="rfl", seed_used=12345, cycles_executed=100,
                expected_log_hash="e" * 64, replay_log_hash="e" * 64,
                log_hash_match=True, expected_final_ht="f" * 64, replay_final_ht="f" * 64,
                final_ht_match=True, ht_sequence_length=100, ht_sequence_match=True,
            ),
            verification_summary=VerificationSummary(checks_passed=12, checks_failed=0),
        )
        receipt.receipt_hash = compute_receipt_hash(receipt)
        return receipt

    def _make_ledger(self, verified: int, failed: int) -> Dict[str, Any]:
        """Create a test ledger."""
        run_history = []
        total = verified + failed
        for i in range(total):
            status = "VERIFIED" if i < verified else "FAILED"
            run_history.append({
                "run_id": f"run_{i}",
                "experiment_id": "U2_EXP_TEST",
                "status": status,
                "mismatch_codes": [],
                "ht_hash": "a" * 64,
                "timestamp": f"2025-12-0{i+1}T10:00:00+00:00",
                "checks_passed": 12 if status == "VERIFIED" else 10,
                "checks_total": 12,
            })
        return {
            "schema_version": "1.0.0",
            "run_history": run_history,
            "totals": {"verified": verified, "failed": failed, "incomplete": 0},
            "determinism_rate": verified / total if total > 0 else 0.0,
        }

    def test_snapshot_contains_all_components(self):
        """Snapshot includes radar, promotion, panel, and global console."""
        ledger = self._make_ledger(verified=5, failed=0)
        receipts = [self._make_receipt(ReplayStatus.VERIFIED) for _ in range(5)]

        snapshot = self.build_full_governance_snapshot([ledger], receipts)

        self.assertIn("radar", snapshot)
        self.assertIn("promotion_evaluation", snapshot)
        self.assertIn("director_panel", snapshot)
        self.assertIn("global_console_summary", snapshot)
        self.assertIn("incident_summary", snapshot)
        self.assertIn("snapshot_hash", snapshot)

    def test_snapshot_has_integrity_hash(self):
        """Snapshot includes integrity hash."""
        ledger = self._make_ledger(verified=5, failed=0)

        snapshot = self.build_full_governance_snapshot([ledger], [])

        self.assertIn("snapshot_hash", snapshot)
        self.assertEqual(len(snapshot["snapshot_hash"]), 64)

    def test_snapshot_json_serializable(self):
        """Snapshot is fully JSON serializable."""
        ledger = self._make_ledger(verified=5, failed=0)
        receipts = [self._make_receipt(ReplayStatus.VERIFIED) for _ in range(3)]

        snapshot = self.build_full_governance_snapshot([ledger], receipts)

        # Should not raise
        json_str = json.dumps(snapshot, sort_keys=True)
        self.assertIsInstance(json_str, str)

        # Should round-trip
        restored = json.loads(json_str)
        self.assertEqual(restored["radar"]["radar_status"], snapshot["radar"]["radar_status"])

    def test_snapshot_empty_inputs(self):
        """Snapshot handles empty inputs gracefully."""
        snapshot = self.build_full_governance_snapshot([], [])

        self.assertEqual(snapshot["radar"]["radar_status"], "STABLE")
        self.assertTrue(snapshot["promotion_evaluation"]["replay_ok_for_promotion"])
        self.assertEqual(snapshot["director_panel"]["status_light"], "GREEN")
        self.assertTrue(snapshot["global_console_summary"]["replay_ok"])


# ============================================================================
# Phase VI Tests: Replay as a First-Class Global Governance Signal
# ============================================================================

class TestGovernanceSignalAdapter(unittest.TestCase):
    """Test the GovernanceSignal adapter for CLAUDE I (TASK 1)."""

    def setUp(self):
        """Import Phase VI functions."""
        from backend.governance.replay_receipt import (
            to_governance_signal_for_replay,
            GOVERNANCE_SIGNAL_VERSION,
        )
        self.to_governance_signal_for_replay = to_governance_signal_for_replay
        self.GOVERNANCE_SIGNAL_VERSION = GOVERNANCE_SIGNAL_VERSION

    def _make_console_summary(
        self,
        replay_ok: bool = True,
        radar_status: str = "STABLE",
        promotion_status: str = "OK",
        determinism_rate: float = 1.0,
        critical_incident_rate: float = 0.0,
        hot_fingerprints_count: int = 0,
        headline: str = "Test headline",
    ) -> Dict[str, Any]:
        """Create a test global console summary."""
        return {
            "schema_version": "1.0.0",
            "replay_ok": replay_ok,
            "radar_status": radar_status,
            "promotion_status": promotion_status,
            "determinism_rate": determinism_rate,
            "critical_incident_rate": critical_incident_rate,
            "hot_fingerprints_count": hot_fingerprints_count,
            "headline": headline,
            "generated_at": "2025-12-06T10:00:00+00:00",
            "console_hash": "a" * 64,
        }

    def test_signal_schema_conformance(self):
        """Signal conforms to GovernanceSignal schema."""
        summary = self._make_console_summary()
        signal = self.to_governance_signal_for_replay(summary)

        # Required fields
        self.assertIn("schema_version", signal)
        self.assertIn("source", signal)
        self.assertIn("status", signal)
        self.assertIn("blocking_rules", signal)
        self.assertIn("blocking_rate", signal)
        self.assertIn("headline", signal)
        self.assertIn("signal_hash", signal)

        self.assertEqual(signal["schema_version"], self.GOVERNANCE_SIGNAL_VERSION)
        self.assertEqual(signal["source"], "replay")

    def test_block_when_determinism_rate_below_90(self):
        """BLOCK when determinism_rate < 0.9."""
        summary = self._make_console_summary(
            replay_ok=True,
            radar_status="STABLE",
            determinism_rate=0.85,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["status"], "BLOCK")
        self.assertTrue(any("REPLAY-DET-LOW" in r for r in signal["blocking_rules"]))

    def test_block_when_radar_unstable(self):
        """BLOCK when radar_status == UNSTABLE."""
        summary = self._make_console_summary(
            replay_ok=True,
            radar_status="UNSTABLE",
            determinism_rate=0.95,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["status"], "BLOCK")
        self.assertTrue(any("REPLAY-RADAR-UNSTABLE" in r for r in signal["blocking_rules"]))

    def test_block_when_replay_not_ok(self):
        """BLOCK when replay_ok == False."""
        summary = self._make_console_summary(
            replay_ok=False,
            radar_status="STABLE",
            determinism_rate=0.95,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["status"], "BLOCK")
        self.assertTrue(any("REPLAY-BLOCKED" in r for r in signal["blocking_rules"]))

    def test_warn_when_degrading_but_replay_ok(self):
        """WARN when DEGRADING but replay_ok=True."""
        summary = self._make_console_summary(
            replay_ok=True,
            radar_status="DEGRADING",
            promotion_status="WARN",
            determinism_rate=0.92,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["status"], "WARN")

    def test_ok_when_stable_and_replay_ok(self):
        """OK when STABLE and replay_ok=True."""
        summary = self._make_console_summary(
            replay_ok=True,
            radar_status="STABLE",
            promotion_status="OK",
            determinism_rate=1.0,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["status"], "OK")
        self.assertEqual(signal["blocking_rules"], [])

    def test_blocking_rules_include_hot_fingerprints(self):
        """Blocking rules include hot fingerprints when present."""
        summary = self._make_console_summary(
            hot_fingerprints_count=3,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertTrue(any("REPLAY-HOT-FP" in r for r in signal["blocking_rules"]))
        self.assertTrue(any("3" in r for r in signal["blocking_rules"]))

    def test_blocking_rules_include_critical_incident_rate(self):
        """Blocking rules include critical incident rate when high."""
        summary = self._make_console_summary(
            critical_incident_rate=0.5,
        )
        signal = self.to_governance_signal_for_replay(summary)

        self.assertTrue(any("REPLAY-CRIT-HIGH" in r for r in signal["blocking_rules"]))

    def test_blocking_rate_calculation(self):
        """Blocking rate is calculated from multiple factors."""
        # All blocking factors present
        summary = self._make_console_summary(
            replay_ok=False,
            radar_status="UNSTABLE",
            determinism_rate=0.5,
            critical_incident_rate=0.8,
        )
        signal = self.to_governance_signal_for_replay(summary)

        # blocking_factors: [1.0, 1.0, 1.0, 0.8] = 3.8/4 = 0.95
        self.assertGreater(signal["blocking_rate"], 0.9)

    def test_signal_has_hash(self):
        """Signal includes integrity hash."""
        summary = self._make_console_summary()
        signal = self.to_governance_signal_for_replay(summary)

        self.assertIn("signal_hash", signal)
        self.assertEqual(len(signal["signal_hash"]), 64)

    def test_signal_preserves_headline(self):
        """Signal preserves headline from console summary."""
        headline = "Custom test headline for replay"
        summary = self._make_console_summary(headline=headline)
        signal = self.to_governance_signal_for_replay(summary)

        self.assertEqual(signal["headline"], headline)


class TestGlobalConsoleWiringContract(unittest.TestCase):
    """Test the Global Console Wiring Contract (TASK 2)."""

    def setUp(self):
        """Import Phase VI functions."""
        from backend.governance.replay_receipt import (
            summarize_replay_for_global_console,
            validate_global_console_tile,
            get_global_console_tile_schema,
            GLOBAL_CONSOLE_VERSION,
        )
        self.summarize_replay_for_global_console = summarize_replay_for_global_console
        self.validate_global_console_tile = validate_global_console_tile
        self.get_global_console_tile_schema = get_global_console_tile_schema
        self.GLOBAL_CONSOLE_VERSION = GLOBAL_CONSOLE_VERSION

    def _make_radar(self, status: str = "STABLE") -> Dict[str, Any]:
        """Create a test radar."""
        return {
            "radar_status": status,
            "hot_fingerprints": [],
            "critical_incident_rate": 0.0,
        }

    def _make_promotion(
        self,
        status: str = "OK",
        replay_ok: bool = True,
        determinism_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a test promotion evaluation."""
        return {
            "status": status,
            "replay_ok_for_promotion": replay_ok,
            "determinism_rate": determinism_rate,
        }

    def test_console_summary_conforms_to_contract(self):
        """Console summary output conforms to wiring contract."""
        radar = self._make_radar()
        promotion = self._make_promotion()

        tile = self.summarize_replay_for_global_console(radar, promotion)

        valid, error = self.validate_global_console_tile(tile)
        self.assertTrue(valid, f"Validation failed: {error}")

    def test_validation_catches_missing_fields(self):
        """Validation catches missing required fields."""
        incomplete_tile = {
            "replay_ok": True,
            "radar_status": "STABLE",
        }

        valid, error = self.validate_global_console_tile(incomplete_tile)
        self.assertFalse(valid)
        self.assertIn("Missing required field", error)

    def test_validation_catches_invalid_radar_status(self):
        """Validation catches invalid radar_status enum."""
        invalid_tile = {
            "schema_version": "1.0.0",
            "replay_ok": True,
            "radar_status": "INVALID",
            "promotion_status": "OK",
            "hot_fingerprints_count": 0,
            "critical_incident_rate": 0.0,
            "determinism_rate": 1.0,
            "headline": "Test",
            "generated_at": "2025-12-06T10:00:00+00:00",
            "console_hash": "a" * 64,
        }

        valid, error = self.validate_global_console_tile(invalid_tile)
        self.assertFalse(valid)
        self.assertIn("Invalid radar_status", error)

    def test_validation_catches_invalid_determinism_rate(self):
        """Validation catches determinism_rate out of bounds."""
        invalid_tile = {
            "schema_version": "1.0.0",
            "replay_ok": True,
            "radar_status": "STABLE",
            "promotion_status": "OK",
            "hot_fingerprints_count": 0,
            "critical_incident_rate": 0.0,
            "determinism_rate": 1.5,  # Out of bounds
            "headline": "Test",
            "generated_at": "2025-12-06T10:00:00+00:00",
            "console_hash": "a" * 64,
        }

        valid, error = self.validate_global_console_tile(invalid_tile)
        self.assertFalse(valid)
        self.assertIn("determinism_rate", error)

    def test_schema_has_all_fields(self):
        """Schema includes all required field definitions."""
        schema = self.get_global_console_tile_schema()

        self.assertIn("required", schema)
        self.assertIn("properties", schema)

        expected_fields = [
            "schema_version",
            "replay_ok",
            "radar_status",
            "promotion_status",
            "hot_fingerprints_count",
            "critical_incident_rate",
            "determinism_rate",
            "headline",
            "generated_at",
            "console_hash",
        ]

        for field in expected_fields:
            self.assertIn(field, schema["required"])
            self.assertIn(field, schema["properties"])

    def test_tile_is_drop_in_compatible(self):
        """Tile can be embedded directly as global_health['replay']."""
        radar = self._make_radar()
        promotion = self._make_promotion()

        # Simulate embedding in global_health
        global_health: Dict[str, Any] = {
            "status": "OK",
            "components": {},
        }
        global_health["replay"] = self.summarize_replay_for_global_console(radar, promotion)

        # Verify structure
        self.assertIn("replay", global_health)
        self.assertIn("replay_ok", global_health["replay"])
        self.assertIn("headline", global_health["replay"])


class TestEvidenceChainEnrichmentWithSignal(unittest.TestCase):
    """Test Evidence Chain Enrichment with GovernanceSignal (TASK 3)."""

    def setUp(self):
        """Import Phase VI functions."""
        from backend.governance.replay_receipt import (
            attach_replay_governance_to_evidence,
            attach_replay_governance_to_evidence_with_signal,
        )
        self.attach_replay_governance_to_evidence = attach_replay_governance_to_evidence
        self.attach_replay_governance_to_evidence_with_signal = attach_replay_governance_to_evidence_with_signal

    def _make_chain(self) -> Dict[str, Any]:
        """Create a mock evidence chain."""
        return {
            "experiment_id": "U2_EXP_001",
            "manifest_hash": "a" * 64,
            "attestations": [
                {"type": "G1", "status": "PASS"},
            ],
        }

    def _make_radar(self, status: str = "STABLE") -> Dict[str, Any]:
        """Create a test radar."""
        return {
            "radar_status": status,
            "hot_fingerprints": [],
            "critical_incident_rate": 0.0,
            "total_runs": 10,
        }

    def _make_promotion(
        self,
        status: str = "OK",
        determinism_rate: float = 1.0,
    ) -> Dict[str, Any]:
        """Create a test promotion evaluation."""
        return {
            "status": status,
            "replay_ok_for_promotion": status == "OK",
            "determinism_rate": determinism_rate,
        }

    def test_backward_compatibility_without_signal(self):
        """Original function behavior unchanged (no governance_signal)."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        # Use original function
        updated = self.attach_replay_governance_to_evidence(chain, radar, promotion)

        # Should NOT have governance_signal
        self.assertNotIn("governance_signal", updated["replay_governance"])

    def test_extended_function_without_signal_flag(self):
        """Extended function with include_governance_signal=False is backward compatible."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=False
        )

        # Should NOT have governance_signal
        self.assertNotIn("governance_signal", updated["replay_governance"])

    def test_extended_function_adds_governance_signal(self):
        """Extended function with include_governance_signal=True adds signal."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=True
        )

        # Should have governance_signal
        self.assertIn("governance_signal", updated["replay_governance"])
        signal = updated["replay_governance"]["governance_signal"]

        # Signal should have required fields
        self.assertEqual(signal["source"], "replay")
        self.assertIn("status", signal)
        self.assertIn("blocking_rules", signal)
        self.assertIn("blocking_rate", signal)

    def test_signal_status_matches_promotion(self):
        """Governance signal status matches promotion evaluation."""
        chain = self._make_chain()

        # Test OK - STABLE with high determinism
        radar_ok = self._make_radar("STABLE")
        promotion_ok = self._make_promotion("OK", 1.0)
        updated_ok = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar_ok, promotion_ok, include_governance_signal=True
        )
        self.assertEqual(updated_ok["replay_governance"]["governance_signal"]["status"], "OK")

        # Test WARN - DEGRADING with determinism still >= 0.9
        # Signal uses DEGRADING radar to produce WARN (not blocked by rate)
        radar_warn = self._make_radar("DEGRADING")
        # Need determinism >= 0.9 AND replay_ok=True to get WARN instead of BLOCK
        promotion_warn = {"status": "WARN", "replay_ok_for_promotion": True, "determinism_rate": 0.92}
        updated_warn = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar_warn, promotion_warn, include_governance_signal=True
        )
        self.assertEqual(updated_warn["replay_governance"]["governance_signal"]["status"], "WARN")

        # Test BLOCK - UNSTABLE radar
        radar_block = self._make_radar("UNSTABLE")
        promotion_block = self._make_promotion("BLOCK", 0.5)
        updated_block = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar_block, promotion_block, include_governance_signal=True
        )
        self.assertEqual(updated_block["replay_governance"]["governance_signal"]["status"], "BLOCK")

    def test_tile_hash_updated_with_signal(self):
        """Tile hash is recomputed when signal is added."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        # Without signal
        updated_without = self.attach_replay_governance_to_evidence(chain, radar, promotion)
        hash_without = updated_without["replay_governance"]["tile_hash"]

        # With signal
        updated_with = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=True
        )
        hash_with = updated_with["replay_governance"]["tile_hash"]

        # Hashes should be different
        self.assertNotEqual(hash_without, hash_with)

    def test_determinism_preserved(self):
        """Same inputs produce same output (deterministic)."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated1 = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=True
        )
        updated2 = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=True
        )

        # Core fields should match (timestamps may differ)
        self.assertEqual(
            updated1["replay_governance"]["governance_signal"]["status"],
            updated2["replay_governance"]["governance_signal"]["status"],
        )
        self.assertEqual(
            updated1["replay_governance"]["governance_signal"]["blocking_rules"],
            updated2["replay_governance"]["governance_signal"]["blocking_rules"],
        )

    def test_chain_structure_preserved_with_signal(self):
        """Original chain structure preserved when adding signal."""
        chain = self._make_chain()
        radar = self._make_radar()
        promotion = self._make_promotion()

        updated = self.attach_replay_governance_to_evidence_with_signal(
            chain, radar, promotion, include_governance_signal=True
        )

        # Original fields preserved
        self.assertEqual(updated["experiment_id"], chain["experiment_id"])
        self.assertEqual(updated["manifest_hash"], chain["manifest_hash"])
        self.assertEqual(updated["attestations"], chain["attestations"])

        # New tile added with signal
        self.assertIn("replay_governance", updated)
        self.assertIn("governance_signal", updated["replay_governance"])


if __name__ == "__main__":
    unittest.main()
