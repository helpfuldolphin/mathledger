"""
PHASE-II — NOT USED IN PHASE I

Deterministic Seed Schedule Generation
======================================

This module provides deterministic seed schedule generation for U2 uplift experiments.
All random operations use seeded RNG instances for full reproducibility.

**Determinism Notes:**
    - Seed schedules are computed deterministically from the initial seed.
    - The same initial seed always produces the same schedule.
    - Uses Python's random.Random for isolated RNG state.
"""

from __future__ import annotations

import random
from typing import List


def generate_seed_schedule(initial_seed: int, num_cycles: int) -> List[int]:
    """Generate a deterministic list of seeds for each experiment cycle.

    This function creates a reproducible sequence of seeds that can be used
    to initialize RNG states for each cycle of an experiment. The sequence
    is fully determined by the initial_seed, ensuring reproducibility.

    Args:
        initial_seed: The initial seed value used to generate the schedule.
            Must be a non-negative integer.
        num_cycles: The number of cycles (seeds) to generate.
            Must be a positive integer.

    Returns:
        A list of ``num_cycles`` integer seeds, each in range [0, 2^32 - 1].

    Raises:
        ValueError: If num_cycles is not positive or initial_seed is negative.

    Example:
        >>> seeds = generate_seed_schedule(42, 5)
        >>> len(seeds)
        5
        >>> generate_seed_schedule(42, 5) == generate_seed_schedule(42, 5)
        True

    **Determinism Notes:**
        - Same initial_seed always produces the same schedule.
        - Uses isolated RNG instance (does not affect global random state).
    """
    if num_cycles <= 0:
        raise ValueError(f"num_cycles must be positive, got {num_cycles}")
    if initial_seed < 0:
        raise ValueError(f"initial_seed must be non-negative, got {initial_seed}")

    rng = random.Random(initial_seed)
    return [rng.randint(0, 2**32 - 1) for _ in range(num_cycles)]


def validate_seed_schedule(
    schedule: List[int], initial_seed: int, num_cycles: int
) -> bool:
    """Validate that a seed schedule matches the expected deterministic output.

    This function can be used to verify that a seed schedule was generated
    correctly from the given initial seed.

    Args:
        schedule: The seed schedule to validate.
        initial_seed: The initial seed that should have generated the schedule.
        num_cycles: The expected number of cycles in the schedule.

    Returns:
        True if the schedule matches the deterministic expectation, False otherwise.

    Example:
        >>> schedule = generate_seed_schedule(42, 5)
        >>> validate_seed_schedule(schedule, 42, 5)
        True
        >>> validate_seed_schedule([1, 2, 3], 42, 3)
        False
    """
    if len(schedule) != num_cycles:
        return False

    expected = generate_seed_schedule(initial_seed, num_cycles)
    return schedule == expected
