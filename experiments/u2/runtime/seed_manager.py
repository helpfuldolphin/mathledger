"""
PHASE II — NOT USED IN PHASE I

Seed Manager Module
===================

Provides deterministic seed schedule generation for U2 uplift experiments.
All seeding is instance-scoped with no global state modifications.

This module extracts and encapsulates the seed management logic from the
original experiments/run_uplift_u2.py to enable testability and reuse.

Example:
    >>> schedule = generate_seed_schedule(initial_seed=42, num_cycles=5)
    >>> schedule.cycle_seeds
    [1608637542, 3421126067, 4083286876, 685946936, 1079662871]
    >>> schedule.algorithm
    'random.Random'
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class SeedSchedule:
    """
    Immutable container for a deterministic seed schedule.

    This dataclass captures the complete seed configuration for a U2 experiment,
    enabling reproducibility verification and audit trails.

    Attributes:
        initial_seed: The master seed used to generate the schedule.
        cycle_seeds: List of per-cycle seeds derived from initial_seed.
        algorithm: String identifier for the PRNG algorithm used.

    Example:
        >>> schedule = SeedSchedule(
        ...     initial_seed=42,
        ...     cycle_seeds=[123, 456, 789],
        ...     algorithm="random.Random"
        ... )
        >>> len(schedule.cycle_seeds)
        3
    """

    initial_seed: int
    cycle_seeds: List[int] = field(default_factory=list)
    algorithm: str = "random.Random"

    def __post_init__(self) -> None:
        """Validate schedule invariants."""
        if not isinstance(self.initial_seed, int):
            raise TypeError(f"initial_seed must be int, got {type(self.initial_seed)}")
        if not isinstance(self.cycle_seeds, list):
            raise TypeError(f"cycle_seeds must be list, got {type(self.cycle_seeds)}")
        for i, seed in enumerate(self.cycle_seeds):
            if not isinstance(seed, int):
                raise TypeError(f"cycle_seeds[{i}] must be int, got {type(seed)}")

    def get_seed(self, cycle: int) -> int:
        """
        Get the seed for a specific cycle index.

        Args:
            cycle: Zero-based cycle index.

        Returns:
            The seed value for the requested cycle.

        Raises:
            IndexError: If cycle is out of range.
        """
        if cycle < 0 or cycle >= len(self.cycle_seeds):
            raise IndexError(
                f"Cycle {cycle} out of range [0, {len(self.cycle_seeds)})"
            )
        return self.cycle_seeds[cycle]

    @property
    def num_cycles(self) -> int:
        """Return the number of cycles in this schedule."""
        return len(self.cycle_seeds)


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> SeedSchedule:
    """
    Generate a deterministic seed schedule for experiment cycles.

    This function produces the exact same sequence of seeds as the original
    inline implementation in experiments/run_uplift_u2.py, ensuring backward
    compatibility and reproducibility.

    The algorithm uses Python's random.Random with instance-scoped state
    (no global random.seed() calls) to generate num_cycles seeds in the
    range [0, 2^32 - 1].

    Args:
        initial_seed: The master seed for the PRNG. Must be an integer.
        num_cycles: Number of cycle seeds to generate. Must be non-negative.

    Returns:
        A SeedSchedule containing the initial seed and all cycle seeds.

    Raises:
        TypeError: If initial_seed is not an integer.
        ValueError: If num_cycles is negative.

    Example:
        >>> schedule = generate_seed_schedule(42, 3)
        >>> schedule.initial_seed
        42
        >>> schedule.cycle_seeds
        [1608637542, 3421126067, 4083286876]

    Note:
        This implementation is semantically equivalent to:
        ```python
        rng = random.Random(initial_seed)
        return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]
        ```
    """
    if not isinstance(initial_seed, int):
        raise TypeError(f"initial_seed must be int, got {type(initial_seed)}")
    if num_cycles < 0:
        raise ValueError(f"num_cycles must be non-negative, got {num_cycles}")

    # Use instance-scoped RNG to avoid global state pollution
    rng = random.Random(initial_seed)

    # Generate seeds in range [0, 2^32 - 1] to match original implementation
    cycle_seeds = [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]

    return SeedSchedule(
        initial_seed=initial_seed,
        cycle_seeds=cycle_seeds,
        algorithm="random.Random",
    )


def hash_string(data: str, algorithm: str = "sha256") -> str:
    """
    Compute a cryptographic hash of a string.

    This function provides a deterministic hashing utility for generating
    content-addressable identifiers, manifest hashes, and telemetry roots.

    Args:
        data: The string to hash. Will be encoded as UTF-8.
        algorithm: Hash algorithm to use. Defaults to "sha256".
            Supported: "sha256", "sha384", "sha512", "md5" (not recommended).

    Returns:
        Lowercase hexadecimal digest of the hash.

    Raises:
        ValueError: If the algorithm is not supported.

    Example:
        >>> hash_string("hello world")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
        >>> hash_string("hello world", algorithm="sha256")
        'b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9'
    """
    algorithm = algorithm.lower()

    if algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "sha384":
        hasher = hashlib.sha384()
    elif algorithm == "sha512":
        hasher = hashlib.sha512()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(
            f"Unsupported hash algorithm: {algorithm}. "
            f"Supported: sha256, sha384, sha512, md5"
        )

    hasher.update(data.encode("utf-8"))
    return hasher.hexdigest()


__all__ = [
    "SeedSchedule",
    "generate_seed_schedule",
    "hash_string",
]

