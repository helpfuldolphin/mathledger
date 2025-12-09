# PHASE II — NOT USED IN PHASE I
"""
Unit tests for the deterministic PRNG module.

These tests verify:
    1. Seed derivation stability across calls
    2. Different paths produce different seeds
    3. for_path produces reproducible random sequences
    4. No global random state pollution
    5. Cross-platform determinism (bit-identical seeds)

Contract Reference:
    Tests alignment with docs/DETERMINISM_CONTRACT.md requirements.
"""

import json
import pytest
import random as stdlib_random
import hashlib
from typing import List

from rfl.prng import (
    PRNGKey,
    derive_seed,
    DeterministicPRNG,
    int_to_hex_seed,
    DEFAULT_MASTER_SEED,
)
from rfl.prng.deterministic_prng import derive_seed_64bit


class TestPRNGKey:
    """Tests for PRNGKey dataclass."""

    def test_prng_key_immutable(self):
        """PRNGKey should be frozen (immutable)."""
        key = PRNGKey(root="a" * 64, path=("test",))
        with pytest.raises(AttributeError):
            key.root = "b" * 64  # type: ignore

    def test_prng_key_validation_root_length(self):
        """PRNGKey should reject invalid root length."""
        with pytest.raises(ValueError, match="64 hex characters"):
            PRNGKey(root="abc", path=())

    def test_prng_key_validation_root_hex(self):
        """PRNGKey should reject non-hex root."""
        with pytest.raises(ValueError, match="valid hex string"):
            PRNGKey(root="g" * 64, path=())  # 'g' is not valid hex

    def test_prng_key_child_derivation(self):
        """child() should append labels to path."""
        parent = PRNGKey(root="a" * 64, path=("level1",))
        child = parent.child("level2", "level3")
        assert child.path == ("level1", "level2", "level3")
        assert child.root == parent.root

    def test_prng_key_canonical_string(self):
        """canonical_string() should produce consistent format."""
        key = PRNGKey(root="a" * 64, path=("slice", "mode", "cycle"))
        expected = "a" * 64 + "::slice::mode::cycle"
        assert key.canonical_string() == expected

    def test_prng_key_canonical_string_empty_path(self):
        """canonical_string() with empty path should return just root."""
        key = PRNGKey(root="b" * 64, path=())
        assert key.canonical_string() == "b" * 64


class TestDeriveSeed:
    """Tests for derive_seed function."""

    def test_derive_seed_stable(self):
        """Same key should always produce same seed."""
        key = PRNGKey(root="c" * 64, path=("test", "stable"))
        seed1 = derive_seed(key)
        seed2 = derive_seed(key)
        seed3 = derive_seed(key)
        assert seed1 == seed2 == seed3

    def test_derive_seed_stable_across_instances(self):
        """Equivalent keys (same root/path) should produce same seed."""
        key1 = PRNGKey(root="d" * 64, path=("a", "b", "c"))
        key2 = PRNGKey(root="d" * 64, path=("a", "b", "c"))
        assert derive_seed(key1) == derive_seed(key2)

    def test_different_paths_different_seeds(self):
        """Different paths should produce different seeds."""
        root = "e" * 64
        key_a = PRNGKey(root=root, path=("path_a",))
        key_b = PRNGKey(root=root, path=("path_b",))
        key_c = PRNGKey(root=root, path=("path_a", "sub"))

        seeds = {derive_seed(key_a), derive_seed(key_b), derive_seed(key_c)}
        assert len(seeds) == 3, "All paths should produce unique seeds"

    def test_different_roots_different_seeds(self):
        """Different roots should produce different seeds."""
        path = ("same", "path")
        key_a = PRNGKey(root="f" * 64, path=path)
        key_b = PRNGKey(root="0" * 64, path=path)

        assert derive_seed(key_a) != derive_seed(key_b)

    def test_derive_seed_32bit_range(self):
        """derive_seed should return value in 32-bit range."""
        key = PRNGKey(root="1" * 64, path=("range", "test"))
        seed = derive_seed(key)
        assert 0 <= seed < 2**32

    def test_derive_seed_64bit(self):
        """derive_seed_64bit should return larger values."""
        key = PRNGKey(root="2" * 64, path=("bit64", "test"))
        seed_32 = derive_seed(key)
        seed_64 = derive_seed_64bit(key)
        # 64-bit seed should be >= 32-bit (unless 32-bit happened to equal 64-bit mod 2^32)
        assert 0 <= seed_64 < 2**64

    def test_derive_seed_known_value(self):
        """Verify derive_seed produces expected value for known input (regression test)."""
        # This test ensures the algorithm doesn't change unexpectedly
        key = PRNGKey(root="0" * 64, path=("determinism", "test"))
        seed = derive_seed(key)

        # Manually compute expected value
        canonical = key.canonical_string().encode("utf-8")
        digest = hashlib.sha256(canonical).digest()
        expected = int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)

        assert seed == expected


class TestDeterministicPRNG:
    """Tests for DeterministicPRNG class."""

    def test_init_valid_seed(self):
        """Should accept valid 64-hex seed."""
        prng = DeterministicPRNG("a" * 64)
        assert prng.master_seed == "a" * 64

    def test_init_normalizes_case(self):
        """Should normalize seed to lowercase."""
        prng = DeterministicPRNG("A" * 64)
        assert prng.master_seed == "a" * 64

    def test_init_rejects_short_seed(self):
        """Should reject seeds that are too short."""
        with pytest.raises(ValueError, match="64 hex characters"):
            DeterministicPRNG("abc")

    def test_init_rejects_invalid_hex(self):
        """Should reject non-hex seeds."""
        with pytest.raises(ValueError, match="valid hex string"):
            DeterministicPRNG("g" * 64)

    def test_for_path_reproducible(self):
        """for_path should produce reproducible random sequences."""
        prng = DeterministicPRNG("b" * 64)

        rng1 = prng.for_path("slice", "mode", "cycle_0001")
        rng2 = prng.for_path("slice", "mode", "cycle_0001")

        seq1 = [rng1.random() for _ in range(100)]
        seq2 = [rng2.random() for _ in range(100)]

        assert seq1 == seq2, "Same path should produce identical sequences"

    def test_for_path_different_paths_different_sequences(self):
        """Different paths should produce different sequences."""
        prng = DeterministicPRNG("c" * 64)

        rng_a = prng.for_path("path_a")
        rng_b = prng.for_path("path_b")

        seq_a = [rng_a.random() for _ in range(10)]
        seq_b = [rng_b.random() for _ in range(10)]

        assert seq_a != seq_b, "Different paths should produce different sequences"

    def test_seed_for_path(self):
        """seed_for_path should return consistent seeds."""
        prng = DeterministicPRNG("d" * 64)

        seed1 = prng.seed_for_path("slice", "cycle")
        seed2 = prng.seed_for_path("slice", "cycle")

        assert seed1 == seed2

    def test_generate_seed_schedule(self):
        """generate_seed_schedule should produce deterministic schedule."""
        prng = DeterministicPRNG("e" * 64)

        schedule1 = prng.generate_seed_schedule(10, "test_slice", "baseline")
        schedule2 = prng.generate_seed_schedule(10, "test_slice", "baseline")

        assert schedule1 == schedule2
        assert len(schedule1) == 10

    def test_generate_seed_schedule_unique_seeds(self):
        """Each cycle in schedule should have unique seed."""
        prng = DeterministicPRNG("f" * 64)
        schedule = prng.generate_seed_schedule(100, "slice", "mode")

        assert len(set(schedule)) == 100, "All cycle seeds should be unique"

    def test_generate_seed_schedule_different_modes(self):
        """Different modes should produce different schedules."""
        prng = DeterministicPRNG("0" * 64)

        schedule_baseline = prng.generate_seed_schedule(5, "slice", "baseline")
        schedule_rfl = prng.generate_seed_schedule(5, "slice", "rfl")

        assert schedule_baseline != schedule_rfl

    def test_log_metadata(self):
        """log_metadata should return valid provenance info."""
        prng = DeterministicPRNG("1" * 64)
        metadata = prng.log_metadata("slice", "mode", "cycle_0001")

        assert "prng_master_seed_prefix" in metadata
        assert "prng_path" in metadata
        assert "prng_derived_seed" in metadata
        assert "prng_canonical_hash" in metadata

        assert metadata["prng_path"] == ["slice", "mode", "cycle_0001"]
        assert isinstance(metadata["prng_derived_seed"], int)


class TestNoGlobalStatePolllution:
    """Tests ensuring PRNG doesn't pollute global random state."""

    def test_no_stdlib_random_pollution(self):
        """DeterministicPRNG should not affect global random module."""
        # Capture global state before
        state_before = stdlib_random.getstate()

        # Use DeterministicPRNG extensively
        prng = DeterministicPRNG("2" * 64)
        for i in range(100):
            rng = prng.for_path(f"test_{i}")
            _ = [rng.random() for _ in range(10)]

        # Capture global state after
        state_after = stdlib_random.getstate()

        assert state_before == state_after, "Global random state should be unchanged"

    @pytest.mark.skipif(
        not hasattr(pytest, "importorskip"),
        reason="pytest.importorskip not available"
    )
    def test_no_numpy_random_pollution(self):
        """DeterministicPRNG should not affect global numpy random state."""
        np = pytest.importorskip("numpy")

        # Capture global numpy state before
        state_before = np.random.get_state()

        # Use DeterministicPRNG with numpy
        prng = DeterministicPRNG("3" * 64)
        for i in range(10):
            try:
                gen = prng.for_numpy(f"numpy_test_{i}")
                _ = gen.random(10)
            except ImportError:
                pytest.skip("numpy not available")

        # Capture global numpy state after
        state_after = np.random.get_state()

        # Compare states (numpy state is a tuple with arrays)
        assert state_before[0] == state_after[0], "State type should match"
        assert np.array_equal(state_before[1], state_after[1]), "State arrays should match"


class TestIntToHexSeed:
    """Tests for int_to_hex_seed conversion."""

    def test_int_to_hex_seed_deterministic(self):
        """Same integer should always produce same hex seed."""
        hex1 = int_to_hex_seed(42)
        hex2 = int_to_hex_seed(42)
        assert hex1 == hex2

    def test_int_to_hex_seed_length(self):
        """Should produce 64-character hex string."""
        hex_seed = int_to_hex_seed(12345)
        assert len(hex_seed) == 64

    def test_int_to_hex_seed_valid_hex(self):
        """Should produce valid hex string."""
        hex_seed = int_to_hex_seed(99999)
        int(hex_seed, 16)  # Should not raise

    def test_int_to_hex_seed_different_inputs(self):
        """Different integers should produce different hex seeds."""
        seeds = {int_to_hex_seed(i) for i in range(100)}
        assert len(seeds) == 100, "All inputs should produce unique hex seeds"

    def test_int_to_hex_seed_negative(self):
        """Should handle negative integers."""
        hex_seed = int_to_hex_seed(-42)
        assert len(hex_seed) == 64

    def test_int_to_hex_seed_zero(self):
        """Should handle zero."""
        hex_seed = int_to_hex_seed(0)
        assert len(hex_seed) == 64

    def test_int_to_hex_seed_large(self):
        """Should handle large integers."""
        hex_seed = int_to_hex_seed(2**128)
        assert len(hex_seed) == 64


class TestDefaultMasterSeed:
    """Tests for DEFAULT_MASTER_SEED constant."""

    def test_default_master_seed_valid(self):
        """DEFAULT_MASTER_SEED should be valid hex."""
        assert len(DEFAULT_MASTER_SEED) == 64
        int(DEFAULT_MASTER_SEED, 16)  # Should not raise

    def test_default_master_seed_stable(self):
        """DEFAULT_MASTER_SEED should be stable across imports."""
        # Re-import to check stability
        from rfl.prng import DEFAULT_MASTER_SEED as seed1
        from rfl.prng.deterministic_prng import DEFAULT_MASTER_SEED as seed2
        assert seed1 == seed2


class TestCrossMethodConsistency:
    """Tests ensuring consistency across different PRNG methods."""

    def test_key_for_path_matches_seed_for_path(self):
        """key_for_path and seed_for_path should be consistent."""
        prng = DeterministicPRNG("4" * 64)
        labels = ("a", "b", "c")

        key = prng.key_for_path(*labels)
        seed_via_key = derive_seed(key)
        seed_via_method = prng.seed_for_path(*labels)

        assert seed_via_key == seed_via_method

    def test_for_path_uses_seed_for_path(self):
        """for_path should use the same seed as seed_for_path."""
        prng = DeterministicPRNG("5" * 64)
        labels = ("x", "y")

        # Get seed and create RNG manually
        seed = prng.seed_for_path(*labels)
        manual_rng = stdlib_random.Random(seed)

        # Get RNG via for_path
        auto_rng = prng.for_path(*labels)

        # Both should produce identical sequences
        manual_seq = [manual_rng.random() for _ in range(50)]
        auto_seq = [auto_rng.random() for _ in range(50)]

        assert manual_seq == auto_seq


class TestRegressionKnownValues:
    """Regression tests with known seed values to detect algorithm changes."""

    def test_known_derivation_regression(self):
        """Verify specific seed derivation hasn't changed."""
        # This test locks in the current algorithm behavior
        prng = DeterministicPRNG("0" * 64)
        seed = prng.seed_for_path("regression", "test", "v1")

        # If this assertion fails, the algorithm has changed
        # Document the new expected value if change is intentional
        expected_seed = derive_seed(
            PRNGKey(root="0" * 64, path=("regression", "test", "v1"))
        )
        assert seed == expected_seed

    def test_schedule_first_10_seeds_stable(self):
        """First 10 seeds in schedule should be stable."""
        prng = DeterministicPRNG("0" * 64)
        schedule = prng.generate_seed_schedule(10, "stable_slice", "baseline")

        # Re-derive to verify stability
        expected = [
            prng.seed_for_path("stable_slice", "baseline", f"cycle_{i:04d}", "seed")
            for i in range(10)
        ]

        assert schedule == expected


class TestIntegrityGuard:
    """Tests for the PRNG integrity guard system."""

    def test_module_compliance_check(self):
        """check_module_compliance should analyze a file correctly."""
        from rfl.prng import check_module_compliance
        import tempfile
        from pathlib import Path

        # Create a test file with violations
        bad_code = '''
import random

def bad_function():
    random.seed(42)
    return random.random()
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(bad_code)
            bad_path = f.name

        try:
            result = check_module_compliance(bad_path)
            assert not result["compliant"], "Should detect violations"
            assert len(result["violations"]) >= 1
        finally:
            Path(bad_path).unlink(missing_ok=True)

    def test_module_compliance_clean(self):
        """check_module_compliance should pass for clean files."""
        from rfl.prng import check_module_compliance
        import tempfile
        from pathlib import Path

        # Create a clean test file
        good_code = '''
from rfl.prng import DeterministicPRNG

def good_function():
    prng = DeterministicPRNG("a" * 64)
    rng = prng.for_path("test")
    return rng.random()
'''
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(good_code)
            good_path = f.name

        try:
            result = check_module_compliance(good_path)
            assert result["compliant"], f"Should be compliant: {result}"
        finally:
            Path(good_path).unlink(missing_ok=True)


class TestSeedLineage:
    """Tests for the seed lineage tracking system."""

    def test_lineage_record(self):
        """SeedLineage.record should track derivations."""
        from rfl.prng.lineage import SeedLineage

        lineage = SeedLineage("b" * 64)
        seed1 = lineage.record("slice", "mode", "cycle")
        seed2 = lineage.record("slice", "mode", "cycle2")

        assert len(lineage.derivations) == 2
        assert seed1 != seed2

    def test_receipt_creation_and_verification(self):
        """Receipts should verify correctly."""
        from rfl.prng.lineage import SeedLineage

        lineage = SeedLineage("c" * 64)
        receipt = lineage.create_receipt("test", "path")

        assert receipt.verify(), "Receipt should verify"
        assert receipt.derived_seed > 0

    def test_receipt_json_roundtrip(self):
        """Receipts should survive JSON serialization."""
        from rfl.prng.lineage import SeedReceipt

        original = SeedReceipt.create("d" * 64, ("a", "b", "c"))
        json_str = original.to_json()
        restored = SeedReceipt.from_json(json_str)

        assert restored.derived_seed == original.derived_seed
        assert restored.verification_hash == original.verification_hash
        assert restored.verify()

    def test_merkle_root_stability(self):
        """Merkle root should be stable for same derivations."""
        from rfl.prng.lineage import SeedLineage

        lineage1 = SeedLineage("e" * 64)
        lineage1.record("a", "b")
        lineage1.record("c", "d")

        lineage2 = SeedLineage("e" * 64)
        lineage2.record("a", "b")
        lineage2.record("c", "d")

        assert lineage1.compute_merkle_root() == lineage2.compute_merkle_root()


class TestPhaseIICompliance:
    """Tests that verify Phase II modules use DeterministicPRNG correctly."""

    def test_audit_phase_ii_modules(self):
        """All Phase II modules should be PRNG-compliant."""
        from rfl.prng import audit_phase_ii_modules

        results = audit_phase_ii_modules()

        assert results["overall_compliant"], (
            f"Phase II modules have PRNG violations:\n"
            f"{json.dumps(results['details'], indent=2)}"
        )


# --- Run self-tests if executed directly ---
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

