"""
PHASE II — NOT USED IN PHASE I

Unit tests for experiments.u2.runtime.seed_manager module.

These tests verify:
    - Deterministic seed schedule generation
    - Backward compatibility with original implementation
    - Hash function correctness
    - Input validation
"""

import unittest
import random

from experiments.u2.runtime.seed_manager import (
    SeedSchedule,
    generate_seed_schedule,
    hash_string,
)


class TestSeedSchedule(unittest.TestCase):
    """Tests for the SeedSchedule dataclass."""

    def test_create_valid_schedule(self) -> None:
        """Test creating a valid SeedSchedule."""
        schedule = SeedSchedule(
            initial_seed=42,
            cycle_seeds=[100, 200, 300],
            algorithm="random.Random",
        )
        self.assertEqual(schedule.initial_seed, 42)
        self.assertEqual(schedule.cycle_seeds, [100, 200, 300])
        self.assertEqual(schedule.algorithm, "random.Random")

    def test_num_cycles_property(self) -> None:
        """Test the num_cycles property."""
        schedule = SeedSchedule(
            initial_seed=42,
            cycle_seeds=[1, 2, 3, 4, 5],
        )
        self.assertEqual(schedule.num_cycles, 5)

    def test_get_seed_valid_index(self) -> None:
        """Test get_seed with valid indices."""
        schedule = SeedSchedule(
            initial_seed=42,
            cycle_seeds=[100, 200, 300],
        )
        self.assertEqual(schedule.get_seed(0), 100)
        self.assertEqual(schedule.get_seed(1), 200)
        self.assertEqual(schedule.get_seed(2), 300)

    def test_get_seed_out_of_range(self) -> None:
        """Test get_seed raises IndexError for out-of-range indices."""
        schedule = SeedSchedule(
            initial_seed=42,
            cycle_seeds=[100, 200, 300],
        )
        with self.assertRaises(IndexError):
            schedule.get_seed(3)
        with self.assertRaises(IndexError):
            schedule.get_seed(-1)

    def test_immutability(self) -> None:
        """Test that SeedSchedule is immutable (frozen)."""
        schedule = SeedSchedule(
            initial_seed=42,
            cycle_seeds=[100, 200],
        )
        with self.assertRaises(AttributeError):
            schedule.initial_seed = 99

    def test_invalid_initial_seed_type(self) -> None:
        """Test that non-integer initial_seed raises TypeError."""
        with self.assertRaises(TypeError):
            SeedSchedule(
                initial_seed="42",  # type: ignore
                cycle_seeds=[100],
            )

    def test_invalid_cycle_seeds_type(self) -> None:
        """Test that non-list cycle_seeds raises TypeError."""
        with self.assertRaises(TypeError):
            SeedSchedule(
                initial_seed=42,
                cycle_seeds=(100, 200),  # type: ignore - tuple not list
            )


class TestGenerateSeedSchedule(unittest.TestCase):
    """Tests for the generate_seed_schedule function."""

    def test_basic_generation(self) -> None:
        """Test basic seed schedule generation."""
        schedule = generate_seed_schedule(42, 5)
        self.assertEqual(schedule.initial_seed, 42)
        self.assertEqual(len(schedule.cycle_seeds), 5)
        self.assertEqual(schedule.algorithm, "random.Random")

    def test_determinism(self) -> None:
        """Test that same inputs produce identical outputs."""
        schedule1 = generate_seed_schedule(42, 10)
        schedule2 = generate_seed_schedule(42, 10)
        self.assertEqual(schedule1.cycle_seeds, schedule2.cycle_seeds)

    def test_backward_compatibility(self) -> None:
        """
        Test that output matches the original inline implementation.

        Original code:
            rng = random.Random(initial_seed)
            return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]
        """
        initial_seed = 42
        num_cycles = 10

        # Generate using new implementation
        schedule = generate_seed_schedule(initial_seed, num_cycles)

        # Generate using original inline implementation
        rng = random.Random(initial_seed)
        original_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]

        # Must be identical
        self.assertEqual(schedule.cycle_seeds, original_seeds)

    def test_different_seeds_produce_different_schedules(self) -> None:
        """Test that different initial seeds produce different schedules."""
        schedule1 = generate_seed_schedule(42, 5)
        schedule2 = generate_seed_schedule(43, 5)
        self.assertNotEqual(schedule1.cycle_seeds, schedule2.cycle_seeds)

    def test_zero_cycles(self) -> None:
        """Test generating zero cycles produces empty schedule."""
        schedule = generate_seed_schedule(42, 0)
        self.assertEqual(schedule.cycle_seeds, [])
        self.assertEqual(schedule.num_cycles, 0)

    def test_negative_cycles_raises(self) -> None:
        """Test that negative num_cycles raises ValueError."""
        with self.assertRaises(ValueError):
            generate_seed_schedule(42, -1)

    def test_non_integer_seed_raises(self) -> None:
        """Test that non-integer initial_seed raises TypeError."""
        with self.assertRaises(TypeError):
            generate_seed_schedule("42", 5)  # type: ignore

    def test_large_cycle_count(self) -> None:
        """Test generation with large cycle count."""
        schedule = generate_seed_schedule(42, 1000)
        self.assertEqual(len(schedule.cycle_seeds), 1000)
        # All seeds should be in valid range
        for seed in schedule.cycle_seeds:
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 2**32)

    def test_seeds_in_valid_range(self) -> None:
        """Test that all generated seeds are in [0, 2^32 - 1]."""
        schedule = generate_seed_schedule(12345, 100)
        for seed in schedule.cycle_seeds:
            self.assertIsInstance(seed, int)
            self.assertGreaterEqual(seed, 0)
            self.assertLessEqual(seed, 2**32 - 1)


class TestHashString(unittest.TestCase):
    """Tests for the hash_string function."""

    def test_sha256_default(self) -> None:
        """Test default SHA256 hashing."""
        result = hash_string("hello world")
        # Known SHA256 hash of "hello world"
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        self.assertEqual(result, expected)

    def test_sha256_explicit(self) -> None:
        """Test explicit SHA256 algorithm."""
        result = hash_string("hello world", algorithm="sha256")
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        self.assertEqual(result, expected)

    def test_sha384(self) -> None:
        """Test SHA384 hashing."""
        result = hash_string("hello world", algorithm="sha384")
        self.assertEqual(len(result), 96)  # SHA384 produces 96 hex chars

    def test_sha512(self) -> None:
        """Test SHA512 hashing."""
        result = hash_string("hello world", algorithm="sha512")
        self.assertEqual(len(result), 128)  # SHA512 produces 128 hex chars

    def test_determinism(self) -> None:
        """Test that same input produces same output."""
        result1 = hash_string("test data")
        result2 = hash_string("test data")
        self.assertEqual(result1, result2)

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Test that different inputs produce different hashes."""
        result1 = hash_string("input1")
        result2 = hash_string("input2")
        self.assertNotEqual(result1, result2)

    def test_empty_string(self) -> None:
        """Test hashing empty string."""
        result = hash_string("")
        # Known SHA256 hash of empty string
        expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        self.assertEqual(result, expected)

    def test_unicode_string(self) -> None:
        """Test hashing Unicode string."""
        result = hash_string("héllo wörld 🌍")
        self.assertEqual(len(result), 64)  # SHA256 produces 64 hex chars

    def test_unsupported_algorithm_raises(self) -> None:
        """Test that unsupported algorithm raises ValueError."""
        with self.assertRaises(ValueError):
            hash_string("test", algorithm="sha999")

    def test_algorithm_case_insensitive(self) -> None:
        """Test that algorithm name is case-insensitive."""
        result1 = hash_string("test", algorithm="SHA256")
        result2 = hash_string("test", algorithm="sha256")
        self.assertEqual(result1, result2)


class TestDeterminismGuarantees(unittest.TestCase):
    """
    Tests ensuring determinism guarantees required by RFL protocol.

    PHASE II — NOT USED IN PHASE I
    """

    def test_repeated_schedule_generation(self) -> None:
        """Test that repeated calls produce identical results."""
        results = [
            generate_seed_schedule(42, 50).cycle_seeds
            for _ in range(100)
        ]
        self.assertTrue(all(r == results[0] for r in results))

    def test_repeated_hashing(self) -> None:
        """Test that repeated hashing produces identical results."""
        data = "test data for determinism verification"
        results = [hash_string(data) for _ in range(100)]
        self.assertTrue(all(r == results[0] for r in results))


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)

