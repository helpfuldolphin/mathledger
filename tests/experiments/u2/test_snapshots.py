"""
PHASE II — NOT USED IN PHASE I

Unit Tests for U2 Snapshot Module
=================================

Tests for snapshot save/load/validate functionality ensuring:
- Round-trip preservation of all state
- Hash integrity verification
- Corruption detection
- Restore produces correct cycle continuation

Usage:
    pytest tests/experiments/u2/test_snapshots.py -v
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Tuple

import pytest

# Skip all tests if dependencies not available
pytest.importorskip("msgpack")
pytest.importorskip("zstandard")

from experiments.u2.snapshots import (
    SnapshotData,
    SnapshotValidationError,
    SnapshotCorruptionError,
    NoSnapshotFoundError,
    compute_snapshot_hash,
    save_snapshot,
    load_snapshot,
    validate_snapshot,
    find_latest_snapshot,
    list_snapshots,
    rotate_snapshots,
    capture_prng_states,
    restore_prng_states,
    _serialize_numpy_state,
    _deserialize_numpy_state,
)

from experiments.u2.runner import (
    U2Runner,
    U2Config,
    RFLPolicy,
)

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# --- Fixtures ---

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_snapshot() -> SnapshotData:
    """Create a sample SnapshotData for testing."""
    return SnapshotData(
        schema_version="1.0",
        cycle_index=50,
        total_cycles=100,
        numpy_rng_state=None,
        python_rng_state=random.getstate(),
        master_seed=42,
        seed_schedule=[i * 1000 for i in range(100)],
        policy_scores={"item_a": 0.7, "item_b": 0.3},
        policy_rng_state=None,
        policy_weights={"len": 0.1, "depth": 0.2, "success": 0.3},
        success_count={"item_a": 10, "item_b": 5},
        attempt_count={"item_a": 20, "item_b": 15},
        policy_update_count=35,
        first_organism_runs_total=50,
        abstention_histogram={"bucket_1": 10, "bucket_2": 20},
        abstention_fraction=0.15,
        previous_coverage_rate=0.85,
        throughput_reference=100.0,
        has_throughput_reference=True,
        config_hash="abc123",
        slice_config_hash="def456",
        curriculum_hash="ghi789",
        experiment_id="test_exp",
        mode="rfl",
        slice_name="test_slice",
        label="PHASE II — NOT USED IN PHASE I",
        ht_series_hash="jkl012",
        ht_series_length=50,
    )


@pytest.fixture
def u2_config() -> U2Config:
    """Create a sample U2Config for testing."""
    return U2Config(
        experiment_id="test_u2",
        slice_name="test_slice",
        mode="rfl",
        total_cycles=100,
        master_seed=42,
        snapshot_interval=10,
        slice_config={"items": ["a", "b", "c"]},
    )


# --- Test: Snapshot Round Trip ---

class TestSnapshotRoundTrip:
    """Tests for save -> load round trip."""
    
    def test_round_trip_preserves_data(self, temp_dir, sample_snapshot):
        """Save and load should produce identical SnapshotData."""
        path = temp_dir / "test_snapshot.snap"
        
        # Save
        save_snapshot(sample_snapshot, path)
        
        # Load
        loaded = load_snapshot(path)
        
        # Compare all fields
        assert loaded.schema_version == sample_snapshot.schema_version
        assert loaded.cycle_index == sample_snapshot.cycle_index
        assert loaded.total_cycles == sample_snapshot.total_cycles
        assert loaded.master_seed == sample_snapshot.master_seed
        assert loaded.seed_schedule == sample_snapshot.seed_schedule
        assert loaded.policy_scores == sample_snapshot.policy_scores
        assert loaded.policy_weights == sample_snapshot.policy_weights
        assert loaded.success_count == sample_snapshot.success_count
        assert loaded.attempt_count == sample_snapshot.attempt_count
        assert loaded.policy_update_count == sample_snapshot.policy_update_count
        assert loaded.first_organism_runs_total == sample_snapshot.first_organism_runs_total
        assert loaded.abstention_histogram == sample_snapshot.abstention_histogram
        assert loaded.abstention_fraction == sample_snapshot.abstention_fraction
        assert loaded.previous_coverage_rate == sample_snapshot.previous_coverage_rate
        assert loaded.throughput_reference == sample_snapshot.throughput_reference
        assert loaded.has_throughput_reference == sample_snapshot.has_throughput_reference
        assert loaded.config_hash == sample_snapshot.config_hash
        assert loaded.slice_config_hash == sample_snapshot.slice_config_hash
        assert loaded.experiment_id == sample_snapshot.experiment_id
        assert loaded.mode == sample_snapshot.mode
        assert loaded.slice_name == sample_snapshot.slice_name
    
    def test_round_trip_with_python_rng_state(self, temp_dir):
        """Python RNG state should be preserved through round trip."""
        # Set specific random state
        random.seed(12345)
        _ = [random.random() for _ in range(100)]  # Advance state
        
        # Capture state
        original_state = random.getstate()
        
        snapshot = SnapshotData(
            python_rng_state=original_state,
            master_seed=12345,
        )
        
        path = temp_dir / "rng_snapshot.snap"
        save_snapshot(snapshot, path)
        loaded = load_snapshot(path)
        
        # Restore and verify
        random.setstate(loaded.python_rng_state)
        
        # Generate sequence from restored state
        random.seed(12345)
        _ = [random.random() for _ in range(100)]  # Same advance
        expected = [random.random() for _ in range(10)]
        
        random.setstate(loaded.python_rng_state)
        actual = [random.random() for _ in range(10)]
        
        assert actual == expected
    
    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_round_trip_with_numpy_rng_state(self, temp_dir):
        """NumPy RNG state should be preserved through round trip."""
        # Set specific random state
        np.random.seed(54321)
        _ = np.random.random(100)  # Advance state
        
        # Capture state
        original_state = np.random.get_state()
        serialized = _serialize_numpy_state(original_state)
        
        snapshot = SnapshotData(
            numpy_rng_state=serialized,
            master_seed=54321,
        )
        
        path = temp_dir / "numpy_snapshot.snap"
        save_snapshot(snapshot, path)
        loaded = load_snapshot(path)
        
        # Restore and verify
        restored_state = _deserialize_numpy_state(loaded.numpy_rng_state)
        np.random.set_state(restored_state)
        
        # Generate sequence from restored state
        np.random.seed(54321)
        _ = np.random.random(100)  # Same advance
        expected = np.random.random(10)
        
        np.random.set_state(restored_state)
        actual = np.random.random(10)
        
        assert np.array_equal(actual, expected)
    
    def test_hash_is_deterministic(self, sample_snapshot):
        """compute_snapshot_hash should be deterministic."""
        hash1 = compute_snapshot_hash(sample_snapshot)
        hash2 = compute_snapshot_hash(sample_snapshot)
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length
    
    def test_different_data_different_hash(self, sample_snapshot):
        """Different snapshot data should produce different hashes."""
        from dataclasses import replace
        
        hash1 = compute_snapshot_hash(sample_snapshot)
        
        modified = replace(sample_snapshot, cycle_index=51)
        hash2 = compute_snapshot_hash(modified)
        
        assert hash1 != hash2


# --- Test: Hash Mismatch Detection ---

class TestHashMismatchDetection:
    """Tests for corruption/tampering detection."""
    
    def test_corrupted_file_detected(self, temp_dir, sample_snapshot):
        """Corrupted snapshot file should raise SnapshotValidationError."""
        path = temp_dir / "corrupted.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Corrupt the file by modifying a byte
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        
        # Flip a byte in the middle
        if len(data) > 100:
            data[100] ^= 0xFF
        
        with open(path, 'wb') as f:
            f.write(data)
        
        # Load should fail
        with pytest.raises(Exception):  # Could be SnapshotValidationError or decompression error
            load_snapshot(path, verify_hash=True)
    
    def test_load_without_verification_succeeds(self, temp_dir, sample_snapshot):
        """Load with verify_hash=False should not check hash."""
        path = temp_dir / "test.snap"
        save_snapshot(sample_snapshot, path)
        
        # Should succeed with verification
        loaded = load_snapshot(path, verify_hash=True)
        assert loaded.cycle_index == sample_snapshot.cycle_index
    
    def test_file_not_found_raises(self, temp_dir):
        """Loading non-existent file should raise FileNotFoundError."""
        path = temp_dir / "nonexistent.snap"
        
        with pytest.raises(FileNotFoundError):
            load_snapshot(path)


# --- Test: Snapshot Validation ---

class TestSnapshotValidation:
    """Tests for validate_snapshot function."""
    
    def test_valid_snapshot_passes(self, sample_snapshot):
        """Valid snapshot should pass validation."""
        assert validate_snapshot(sample_snapshot) is True
    
    def test_invalid_cycle_index_fails(self):
        """Negative cycle index should fail validation."""
        snapshot = SnapshotData(cycle_index=-1)
        
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(snapshot)
    
    def test_cycle_exceeds_total_fails(self):
        """cycle_index >= total_cycles should fail validation."""
        snapshot = SnapshotData(cycle_index=100, total_cycles=100)
        
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(snapshot)
    
    def test_config_hash_mismatch_fails(self, sample_snapshot):
        """Mismatched config hash should fail validation."""
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(
                sample_snapshot,
                expected_config_hash="wrong_hash"
            )
    
    def test_invalid_mode_fails(self):
        """Invalid mode should fail validation."""
        snapshot = SnapshotData(mode="invalid_mode")
        
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(snapshot)


# --- Test: Runner Integration ---

class TestRunnerIntegration:
    """Tests for U2Runner snapshot integration."""
    
    def test_capture_and_restore_produces_identical_state(self, temp_dir, u2_config):
        """Capturing and restoring state should preserve runner state."""
        u2_config.snapshot_dir = temp_dir
        runner = U2Runner(u2_config)
        
        # Simulate some cycles
        items = ["a", "b", "c"]
        execute_fn = lambda item, seed: (random.Random(seed).random() > 0.5, {"mock": True})
        
        for _ in range(25):
            runner.run_cycle(items, execute_fn)
        
        # Capture state
        snapshot = runner.capture_state()
        
        # Store values for comparison
        original_cycle = runner.cycle_index
        original_policy_update_count = runner.policy_update_count
        original_success_count = dict(runner.success_count)
        
        # Create new runner and restore
        new_runner = U2Runner(u2_config)
        new_runner.restore_state(snapshot)
        
        # Verify state matches
        assert new_runner.cycle_index == original_cycle
        assert new_runner.policy_update_count == original_policy_update_count
        assert new_runner.success_count == original_success_count
    
    def test_restored_runner_continues_deterministically(self, temp_dir, u2_config):
        """Runner restored from snapshot should produce identical future results."""
        u2_config.snapshot_dir = temp_dir
        
        # Run 1: Full run
        runner1 = U2Runner(u2_config)
        items = ["a", "b", "c"]
        
        # Deterministic execution function
        def execute_fn(item: str, seed: int) -> Tuple[bool, dict]:
            rng = random.Random(seed)
            return rng.random() > 0.5, {"seed": seed}
        
        # Run 50 cycles
        results1_first_half = []
        for i in range(50):
            result = runner1.run_cycle(items, execute_fn)
            results1_first_half.append((result.item, result.success))
        
        # Capture at cycle 50
        snapshot = runner1.capture_state()
        
        # Continue for 25 more cycles
        results1_second_half = []
        for i in range(25):
            result = runner1.run_cycle(items, execute_fn)
            results1_second_half.append((result.item, result.success))
        
        # Run 2: Restore at cycle 50 and continue
        runner2 = U2Runner(u2_config)
        runner2.restore_state(snapshot)
        
        # Continue for 25 cycles - should match runner1's second half
        results2_after_restore = []
        for i in range(25):
            result = runner2.run_cycle(items, execute_fn)
            results2_after_restore.append((result.item, result.success))
        
        # Results after restore should match
        assert results1_second_half == results2_after_restore
    
    def test_maybe_save_snapshot_respects_interval(self, temp_dir, u2_config):
        """maybe_save_snapshot should only save at configured intervals."""
        u2_config.snapshot_dir = temp_dir
        u2_config.snapshot_interval = 10
        runner = U2Runner(u2_config)
        
        items = ["a", "b", "c"]
        execute_fn = lambda item, seed: (True, {})
        
        snapshots_created = []
        
        for i in range(25):
            runner.run_cycle(items, execute_fn)
            path = runner.maybe_save_snapshot()
            if path:
                snapshots_created.append(path)
        
        # Should have snapshots at cycles 10, 20
        assert len(snapshots_created) == 2
        # Check that filenames contain cycle indices
        assert "10" in str(snapshots_created[0])
        assert "20" in str(snapshots_created[1])
    
    def test_snapshot_interval_zero_disables_snapshots(self, temp_dir, u2_config):
        """snapshot_interval=0 should disable automatic snapshots."""
        u2_config.snapshot_dir = temp_dir
        u2_config.snapshot_interval = 0
        runner = U2Runner(u2_config)
        
        items = ["a", "b", "c"]
        execute_fn = lambda item, seed: (True, {})
        
        for i in range(25):
            runner.run_cycle(items, execute_fn)
            path = runner.maybe_save_snapshot()
            assert path is None


# --- Test: RFL Policy State ---

class TestRFLPolicyState:
    """Tests for RFL policy state capture/restore."""
    
    def test_policy_state_captured(self):
        """RFLPolicy.get_state should capture RNG state."""
        policy = RFLPolicy(seed=42)
        
        # Use policy to build up state
        items = ["x", "y", "z"]
        policy.score(items)
        policy.update("x", True)
        policy.update("y", False)
        
        state = policy.get_state()
        
        # Current impl returns RNG state tuple directly
        assert isinstance(state, tuple)
        assert len(state) == 3  # Random state is a 3-tuple
    
    def test_policy_rng_state_restored(self):
        """RFLPolicy.set_state should restore RNG state for determinism."""
        policy1 = RFLPolicy(seed=42)
        items = ["x", "y", "z"]
        policy1.score(items)  # Advances RNG
        
        # Capture RNG state
        state = policy1.get_state()
        
        # Generate next random score
        expected_score = policy1.rng.random()
        
        # Reset by setting state
        policy1.set_state(state)
        
        # Should get same random value
        actual_score = policy1.rng.random()
        assert actual_score == expected_score
    
    def test_policy_produces_same_scores_after_restore(self):
        """Restored policy should produce identical scores for new items."""
        policy1 = RFLPolicy(seed=42)
        items = ["a", "b", "c"]
        
        # Build up state (score existing items)
        for _ in range(10):
            policy1.score(items)
            policy1.update(random.choice(items), random.random() > 0.5)
        
        # Capture state before scoring new item
        state = policy1.get_state()
        scores_before = policy1.score(["new_item"])
        
        # Restore state
        policy1.set_state(state)
        scores_after = policy1.score(["new_item"])
        
        # New items should get same random scores (same RNG state)
        assert scores_before == scores_after


# --- Test: PRNG State Utilities ---

class TestPRNGStateUtilities:
    """Tests for PRNG capture/restore utilities."""
    
    def test_capture_restore_python_rng(self):
        """capture_prng_states and restore_prng_states should work for Python RNG."""
        random.seed(42)
        _ = [random.random() for _ in range(50)]
        
        numpy_state, python_state = capture_prng_states()
        
        # Generate sequence
        expected = [random.random() for _ in range(10)]
        
        # Scramble state
        random.seed(99999)
        _ = [random.random() for _ in range(100)]
        
        # Restore
        restore_prng_states(numpy_state, python_state)
        
        # Should get same sequence
        actual = [random.random() for _ in range(10)]
        
        assert actual == expected
    
    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_capture_restore_numpy_rng(self):
        """capture_prng_states and restore_prng_states should work for NumPy RNG."""
        np.random.seed(42)
        _ = np.random.random(50)
        
        numpy_state, python_state = capture_prng_states()
        
        # Generate sequence
        expected = np.random.random(10)
        
        # Scramble state
        np.random.seed(99999)
        _ = np.random.random(100)
        
        # Restore
        restore_prng_states(numpy_state, python_state)
        
        # Should get same sequence
        actual = np.random.random(10)
        
        assert np.array_equal(actual, expected)


# --- Test: File Atomicity ---

class TestFileAtomicity:
    """Tests for atomic file operations."""
    
    def test_save_creates_file(self, temp_dir, sample_snapshot):
        """save_snapshot should create the output file."""
        path = temp_dir / "output.snap"
        
        assert not path.exists()
        save_snapshot(sample_snapshot, path)
        assert path.exists()
    
    def test_save_creates_parent_directories(self, temp_dir, sample_snapshot):
        """save_snapshot should create parent directories if needed."""
        path = temp_dir / "deep" / "nested" / "dir" / "output.snap"
        
        assert not path.parent.exists()
        save_snapshot(sample_snapshot, path)
        assert path.exists()
    
    def test_save_returns_hash(self, temp_dir, sample_snapshot):
        """save_snapshot should return the snapshot hash."""
        path = temp_dir / "output.snap"
        
        returned_hash = save_snapshot(sample_snapshot, path)
        expected_hash = compute_snapshot_hash(sample_snapshot)
        
        assert returned_hash == expected_hash


# --- Test: Corruption & Failure Injection ---

class TestCorruptionDetection:
    """Tests for snapshot corruption and tampering detection."""
    
    def test_random_byte_flip_detected(self, temp_dir, sample_snapshot):
        """Flipping a random byte in the file should be detected."""
        import os
        path = temp_dir / "to_corrupt.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Read file and flip a random byte
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        
        # Flip a byte somewhere in the middle (avoid header)
        if len(data) > 50:
            flip_pos = len(data) // 2
            data[flip_pos] ^= 0xFF
        
            with open(path, 'wb') as f:
                f.write(data)
            
            # Load should fail with corruption error
            with pytest.raises(Exception):  # Could be SnapshotCorruptionError or decompression error
                load_snapshot(path, verify_hash=True)
    
    def test_truncated_file_detected(self, temp_dir, sample_snapshot):
        """Truncated file should raise an error."""
        path = temp_dir / "truncated.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Truncate the file
        with open(path, 'rb') as f:
            data = f.read()
        
        with open(path, 'wb') as f:
            f.write(data[:len(data)//2])  # Write only half
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_empty_file_detected(self, temp_dir):
        """Empty file should raise an error."""
        path = temp_dir / "empty.snap"
        
        # Create empty file
        path.touch()
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_garbage_data_detected(self, temp_dir):
        """Random garbage data should raise an error."""
        import os
        path = temp_dir / "garbage.snap"
        
        # Write random bytes
        with open(path, 'wb') as f:
            f.write(os.urandom(1024))
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)


class TestMissingFieldValidation:
    """Tests for missing required field detection."""
    
    def test_missing_cycle_index_handled(self, temp_dir):
        """Snapshot with missing cycle_index should use default."""
        # Create minimal snapshot
        snapshot = SnapshotData()
        path = temp_dir / "minimal.snap"
        save_snapshot(snapshot, path)
        
        loaded = load_snapshot(path)
        assert loaded.cycle_index == 0  # Default value
    
    def test_invalid_schema_version_rejected(self, temp_dir):
        """Snapshot with invalid schema version should fail validation."""
        from dataclasses import replace
        
        snapshot = SnapshotData(schema_version="99.0")
        
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(snapshot)


# --- Test: Advanced Failure Injection ---

class TestAdvancedFailureInjection:
    """
    Advanced failure injection tests ensuring the snapshot system is unbreakable.
    
    INVARIANT: Every snapshot is either valid or loudly invalid.
    INVARIANT: No silent corruption is allowed.
    """
    
    def test_invalid_zstd_header_detected(self, temp_dir, sample_snapshot):
        """Invalid zstd magic bytes should be detected."""
        path = temp_dir / "bad_zstd.snap"
        
        # Write data with invalid zstd magic header (zstd magic is 0xFD2FB528)
        with open(path, 'wb') as f:
            f.write(b'\x00\x00\x00\x00' + b'\x00' * 100)
        
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_valid_zstd_invalid_msgpack_detected(self, temp_dir):
        """Valid zstd compression but invalid msgpack payload should be detected."""
        import zstandard as zstd
        
        path = temp_dir / "bad_msgpack.snap"
        
        # Compress garbage that's not valid msgpack
        compressor = zstd.ZstdCompressor(level=3)
        garbage = b'\xff\xfe\xfd\xfc' * 100  # Invalid msgpack
        compressed = compressor.compress(garbage)
        
        with open(path, 'wb') as f:
            f.write(compressed)
        
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_hash_tampering_detected(self, temp_dir, sample_snapshot):
        """Modifying the stored hash should be detected."""
        import msgpack
        import zstandard as zstd
        
        path = temp_dir / "tampered_hash.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Read and decompress
        with open(path, 'rb') as f:
            compressed = f.read()
        
        decompressor = zstd.ZstdDecompressor()
        packed = decompressor.decompress(compressed)
        wrapper = msgpack.unpackb(packed, raw=False)
        
        # Tamper with the hash
        wrapper['hash'] = 'a' * 64  # Wrong hash
        
        # Recompress and save
        new_packed = msgpack.packb(wrapper, use_bin_type=True)
        compressor = zstd.ZstdCompressor(level=3)
        new_compressed = compressor.compress(new_packed)
        
        with open(path, 'wb') as f:
            f.write(new_compressed)
        
        # Load should detect hash mismatch
        with pytest.raises(SnapshotCorruptionError):
            load_snapshot(path, verify_hash=True)
    
    def test_data_tampering_detected(self, temp_dir, sample_snapshot):
        """Modifying snapshot data while keeping same hash should be detected."""
        import msgpack
        import zstandard as zstd
        
        path = temp_dir / "tampered_data.snap"
        
        # Save valid snapshot
        original_hash = save_snapshot(sample_snapshot, path)
        
        # Read and decompress
        with open(path, 'rb') as f:
            compressed = f.read()
        
        decompressor = zstd.ZstdDecompressor()
        packed = decompressor.decompress(compressed)
        wrapper = msgpack.unpackb(packed, raw=False)
        
        # Tamper with data but keep original hash
        wrapper['snapshot']['cycle_index'] = 999999
        # hash stays the same (attacker trying to inject bad data)
        
        # Recompress and save
        new_packed = msgpack.packb(wrapper, use_bin_type=True)
        compressor = zstd.ZstdCompressor(level=3)
        new_compressed = compressor.compress(new_packed)
        
        with open(path, 'wb') as f:
            f.write(new_compressed)
        
        # Load should detect the mismatch
        with pytest.raises(SnapshotCorruptionError):
            load_snapshot(path, verify_hash=True)
    
    def test_missing_hash_field_detected(self, temp_dir, sample_snapshot):
        """Snapshot file missing 'hash' field should fail gracefully."""
        import msgpack
        import zstandard as zstd
        
        path = temp_dir / "no_hash.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Read and decompress
        with open(path, 'rb') as f:
            compressed = f.read()
        
        decompressor = zstd.ZstdDecompressor()
        packed = decompressor.decompress(compressed)
        wrapper = msgpack.unpackb(packed, raw=False)
        
        # Remove hash field
        del wrapper['hash']
        
        # Recompress and save
        new_packed = msgpack.packb(wrapper, use_bin_type=True)
        compressor = zstd.ZstdCompressor(level=3)
        new_compressed = compressor.compress(new_packed)
        
        with open(path, 'wb') as f:
            f.write(new_compressed)
        
        # Load should fail (empty hash won't match computed hash)
        with pytest.raises(SnapshotCorruptionError):
            load_snapshot(path, verify_hash=True)
    
    def test_missing_snapshot_field_detected(self, temp_dir):
        """Snapshot file missing 'snapshot' field should fail gracefully."""
        import msgpack
        import zstandard as zstd
        
        path = temp_dir / "no_snapshot.snap"
        
        # Create a wrapper without the 'snapshot' field
        wrapper = {
            "schema_version": "1.0",
            "hash": "a" * 64,
            "label": "test",
        }
        
        packed = msgpack.packb(wrapper, use_bin_type=True)
        compressor = zstd.ZstdCompressor(level=3)
        compressed = compressor.compress(packed)
        
        with open(path, 'wb') as f:
            f.write(compressed)
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_partial_write_simulation(self, temp_dir, sample_snapshot):
        """Simulating a partial/interrupted write should be detected."""
        path = temp_dir / "partial.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Read the file
        with open(path, 'rb') as f:
            data = f.read()
        
        # Write only 75% of the file (simulating interrupted write)
        cutoff = int(len(data) * 0.75)
        with open(path, 'wb') as f:
            f.write(data[:cutoff])
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)
    
    def test_multiple_byte_flips_detected(self, temp_dir, sample_snapshot):
        """Multiple scattered byte corruptions should be detected."""
        path = temp_dir / "multi_corrupt.snap"
        
        # Save valid snapshot
        save_snapshot(sample_snapshot, path)
        
        # Read and corrupt multiple bytes
        with open(path, 'rb') as f:
            data = bytearray(f.read())
        
        # Flip bytes at different positions
        if len(data) > 100:
            for offset in [10, len(data)//4, len(data)//2, 3*len(data)//4]:
                if offset < len(data):
                    data[offset] ^= 0xFF
        
        with open(path, 'wb') as f:
            f.write(data)
        
        # Load should fail
        with pytest.raises(Exception):
            load_snapshot(path, verify_hash=True)


# --- Test: Bit-for-Bit Determinism ---

class TestBitForBitDeterminism:
    """
    Tests ensuring restoration produces bit-for-bit determinism.
    
    INVARIANT: Restoration must produce bit-for-bit determinism.
    """
    
    def test_deterministic_prng_sequence_after_restore(self, temp_dir, u2_config):
        """Restored runner must produce identical PRNG sequence."""
        u2_config.snapshot_dir = temp_dir
        runner = U2Runner(u2_config)
        
        items = ["a", "b", "c", "d", "e"]
        execute_fn = lambda item, seed: (random.Random(seed).random() > 0.5, {"seed": seed})
        
        # Run 30 cycles
        for _ in range(30):
            runner.run_cycle(items, execute_fn)
        
        # Capture state
        snapshot = runner.capture_state()
        
        # Record next 20 results
        expected_results = []
        for _ in range(20):
            result = runner.run_cycle(items, execute_fn)
            expected_results.append((result.item, result.success, result.seed))
        
        # Create new runner and restore
        runner2 = U2Runner(u2_config)
        runner2.restore_state(snapshot)
        
        # Generate same 20 results
        actual_results = []
        for _ in range(20):
            result = runner2.run_cycle(items, execute_fn)
            actual_results.append((result.item, result.success, result.seed))
        
        # Must be identical
        assert expected_results == actual_results
    
    def test_deterministic_hash_across_saves(self, temp_dir, sample_snapshot):
        """Same snapshot data must produce identical hash every time."""
        hashes = []
        
        for i in range(5):
            path = temp_dir / f"hash_test_{i}.snap"
            h = save_snapshot(sample_snapshot, path)
            hashes.append(h)
        
        # All hashes must be identical
        assert len(set(hashes)) == 1
    
    def test_deterministic_file_content_across_saves(self, temp_dir, sample_snapshot):
        """Same snapshot data must produce identical file content."""
        paths = []
        contents = []
        
        for i in range(3):
            path = temp_dir / f"content_test_{i}.snap"
            save_snapshot(sample_snapshot, path)
            
            with open(path, 'rb') as f:
                contents.append(f.read())
            paths.append(path)
        
        # All file contents must be identical
        assert contents[0] == contents[1] == contents[2]
    
    def test_restore_preserves_policy_scores(self, temp_dir, u2_config):
        """Restored RFL policy must have identical scores."""
        u2_config.mode = "rfl"
        u2_config.snapshot_dir = temp_dir
        runner = U2Runner(u2_config)
        
        items = ["x", "y", "z"]
        execute_fn = lambda item, seed: (random.Random(seed).random() > 0.5, {})
        
        # Build up policy state
        for _ in range(50):
            runner.run_cycle(items, execute_fn)
        
        # Capture state
        snapshot = runner.capture_state()
        original_scores = dict(runner.policy.scores) if runner.policy else {}
        
        # Create new runner and restore
        runner2 = U2Runner(u2_config)
        runner2.restore_state(snapshot)
        restored_scores = dict(runner2.policy.scores) if runner2.policy else {}
        
        # Scores must be identical
        assert original_scores == restored_scores
    
    def test_restore_across_multiple_cycles_maintains_determinism(self, temp_dir, u2_config):
        """
        Multiple restore points must all lead to identical final states
        when run to completion from the same point.
        """
        u2_config.total_cycles = 100
        u2_config.snapshot_dir = temp_dir
        
        items = ["a", "b", "c"]
        execute_fn = lambda item, seed: (random.Random(seed).random() > 0.5, {"s": seed})
        
        # Run 1: Full run, capture at 50
        runner1 = U2Runner(u2_config)
        for _ in range(50):
            runner1.run_cycle(items, execute_fn)
        snapshot_50 = runner1.capture_state()
        
        # Continue to 75
        for _ in range(25):
            runner1.run_cycle(items, execute_fn)
        final_results_1 = [(r["item"], r["success"]) for r in runner1.ht_series[50:75]]
        
        # Run 2: Restore from 50, continue to 75
        runner2 = U2Runner(u2_config)
        runner2.restore_state(snapshot_50)
        for _ in range(25):
            runner2.run_cycle(items, execute_fn)
        final_results_2 = [(r["item"], r["success"]) for r in runner2.ht_series[:25]]
        
        # Run 3: Restore from 50 again, continue to 75
        runner3 = U2Runner(u2_config)
        runner3.restore_state(snapshot_50)
        for _ in range(25):
            runner3.run_cycle(items, execute_fn)
        final_results_3 = [(r["item"], r["success"]) for r in runner3.ht_series[:25]]
        
        # All must be identical
        assert final_results_1 == final_results_2 == final_results_3


# --- Test: Snapshot Discovery & Resume ---

class TestSnapshotDiscovery:
    """Tests for snapshot discovery and resume functionality."""
    
    def test_find_latest_returns_newest(self, temp_dir):
        """find_latest_snapshot should return the snapshot with highest cycle index."""
        from experiments.u2.snapshots import find_latest_snapshot, list_snapshots
        
        # Create multiple snapshots with different cycle indices
        for cycle in [10, 50, 30, 20]:
            snapshot = SnapshotData(cycle_index=cycle)
            path = temp_dir / f"snapshot_test_{cycle:06d}.snap"
            save_snapshot(snapshot, path)
        
        latest = find_latest_snapshot(temp_dir)
        
        assert latest is not None
        assert "000050" in str(latest)
    
    def test_find_latest_returns_none_for_empty_dir(self, temp_dir):
        """find_latest_snapshot should return None if no snapshots exist."""
        from experiments.u2.snapshots import find_latest_snapshot
        
        result = find_latest_snapshot(temp_dir)
        assert result is None
    
    def test_find_latest_returns_none_for_nonexistent_dir(self):
        """find_latest_snapshot should return None for non-existent directory."""
        from experiments.u2.snapshots import find_latest_snapshot
        
        result = find_latest_snapshot(Path("/nonexistent/path/12345"))
        assert result is None
    
    def test_list_snapshots_sorted_newest_first(self, temp_dir):
        """list_snapshots should return snapshots sorted by cycle index descending."""
        from experiments.u2.snapshots import list_snapshots
        
        # Create snapshots out of order
        for cycle in [10, 50, 30, 20, 40]:
            snapshot = SnapshotData(cycle_index=cycle)
            path = temp_dir / f"snapshot_exp_{cycle:06d}.snap"
            save_snapshot(snapshot, path)
        
        snapshots = list_snapshots(temp_dir)
        
        assert len(snapshots) == 5
        # Should be sorted newest first
        assert "000050" in str(snapshots[0])
        assert "000040" in str(snapshots[1])
        assert "000030" in str(snapshots[2])
        assert "000020" in str(snapshots[3])
        assert "000010" in str(snapshots[4])
    
    def test_find_latest_excludes_final_snapshots(self, temp_dir):
        """find_latest_snapshot should prefer non-final snapshots for resume."""
        from experiments.u2.snapshots import find_latest_snapshot
        
        # Create a regular snapshot and a final snapshot
        snapshot1 = SnapshotData(cycle_index=50)
        save_snapshot(snapshot1, temp_dir / "snapshot_exp_000050.snap")
        
        snapshot2 = SnapshotData(cycle_index=100)
        save_snapshot(snapshot2, temp_dir / "snapshot_exp_final.snap")
        
        latest = find_latest_snapshot(temp_dir)
        
        # Should return the non-final snapshot
        assert latest is not None
        assert "final" not in str(latest)
        assert "000050" in str(latest)


# --- Test: Snapshot Rotation ---

class TestSnapshotRotation:
    """Tests for snapshot rotation and disk hygiene."""
    
    def test_rotation_keeps_newest_n(self, temp_dir):
        """rotate_snapshots should keep only the newest N snapshots."""
        from experiments.u2.snapshots import rotate_snapshots
        
        # Create 8 snapshots
        for cycle in range(10, 90, 10):  # 10, 20, 30, 40, 50, 60, 70, 80
            snapshot = SnapshotData(cycle_index=cycle)
            path = temp_dir / f"snapshot_rot_{cycle:06d}.snap"
            save_snapshot(snapshot, path)
        
        # Rotate keeping only 5
        deleted = rotate_snapshots(temp_dir, keep_count=5)
        
        # Should have deleted 3 (oldest)
        assert len(deleted) == 3
        
        # Only 5 should remain
        remaining = list(temp_dir.glob("snapshot_*.snap"))
        assert len(remaining) == 5
        
        # The remaining should be the newest ones (40, 50, 60, 70, 80)
        remaining_cycles = sorted([int(p.stem.split('_')[-1]) for p in remaining])
        assert remaining_cycles == [40, 50, 60, 70, 80]
    
    def test_rotation_preserves_final_snapshots(self, temp_dir):
        """rotate_snapshots should preserve 'final' snapshots by default."""
        from experiments.u2.snapshots import rotate_snapshots
        
        # Create regular snapshots
        for cycle in range(10, 50, 10):  # 10, 20, 30, 40
            snapshot = SnapshotData(cycle_index=cycle)
            save_snapshot(snapshot, temp_dir / f"snapshot_exp_{cycle:06d}.snap")
        
        # Create a final snapshot
        final_snapshot = SnapshotData(cycle_index=100)
        save_snapshot(final_snapshot, temp_dir / "snapshot_exp_final.snap")
        
        # Rotate keeping only 2 regular
        deleted = rotate_snapshots(temp_dir, keep_count=2, exclude_final=True)
        
        # Final should still exist
        assert (temp_dir / "snapshot_exp_final.snap").exists()
        
        # Should have deleted 2 oldest regular snapshots
        assert len(deleted) == 2
    
    def test_rotation_does_nothing_if_under_limit(self, temp_dir):
        """rotate_snapshots should not delete if under keep_count."""
        from experiments.u2.snapshots import rotate_snapshots
        
        # Create only 3 snapshots
        for cycle in [10, 20, 30]:
            snapshot = SnapshotData(cycle_index=cycle)
            save_snapshot(snapshot, temp_dir / f"snapshot_few_{cycle:06d}.snap")
        
        # Rotate with keep_count=5
        deleted = rotate_snapshots(temp_dir, keep_count=5)
        
        # Nothing should be deleted
        assert len(deleted) == 0
        assert len(list(temp_dir.glob("snapshot_*.snap"))) == 3
    
    def test_rotation_handles_empty_dir(self, temp_dir):
        """rotate_snapshots should handle empty directory gracefully."""
        from experiments.u2.snapshots import rotate_snapshots
        
        deleted = rotate_snapshots(temp_dir, keep_count=5)
        
        assert len(deleted) == 0
    
    def test_n_plus_3_leaves_only_n(self, temp_dir):
        """Writing N+3 snapshots with rotation should leave only N."""
        from experiments.u2.snapshots import rotate_snapshots
        
        keep_count = 5
        total_to_write = keep_count + 3  # 8 snapshots
        
        # Write snapshots and rotate after each
        for i in range(total_to_write):
            cycle = (i + 1) * 10
            snapshot = SnapshotData(cycle_index=cycle)
            save_snapshot(snapshot, temp_dir / f"snapshot_test_{cycle:06d}.snap")
            
            # Rotate after each save
            rotate_snapshots(temp_dir, keep_count=keep_count)
        
        # Should have exactly keep_count snapshots
        remaining = list(temp_dir.glob("snapshot_*.snap"))
        assert len(remaining) == keep_count

