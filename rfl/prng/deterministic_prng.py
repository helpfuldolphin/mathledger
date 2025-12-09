# PHASE II — NOT USED IN PHASE I
"""
Deterministic PRNG with hierarchical seed derivation.

This module provides a stateless, SHA-256-based PRNG framework for
Phase II U2 experiments. All randomness is derived from a master seed
and a hierarchical path of labels, ensuring bit-for-bit reproducibility.

Design Principles:
    1. STATELESS: derive_seed(key) is a pure function of key only.
    2. HIERARCHICAL: Seeds are scoped by path labels (slice, mode, cycle, component).
    3. SHA-256 BASED: Uses cryptographic hash for uniform distribution.
    4. NO GLOBAL STATE: Never calls random.seed() or np.random.seed().

Contract Reference:
    This module implements the deterministic PRNG contract described in
    docs/DETERMINISM_CONTRACT.md and docs/FIRST_ORGANISM_DETERMINISM.md.
    Specifically, it replaces ad-hoc random.Random(seed) usage with a
    disciplined hierarchical derivation chain:

        U2_MASTER_SEED → slice_seed → cycle_seed → component_seed

Example:
    >>> prng = DeterministicPRNG("a" * 64)
    >>> rng1 = prng.for_path("slice_uplift_sparse", "baseline", "cycle_0001")
    >>> rng2 = prng.for_path("slice_uplift_sparse", "baseline", "cycle_0001")
    >>> rng1.random() == rng2.random()  # Always True
    True
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

# Optional numpy support
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# Default master seed (SHA-256 of "MATHLEDGER_U2_DETERMINISTIC_PRNG_V1")
# This provides a stable default for testing; production should use manifest-derived seeds.
DEFAULT_MASTER_SEED = hashlib.sha256(
    b"MATHLEDGER_U2_DETERMINISTIC_PRNG_V1"
).hexdigest()


def int_to_hex_seed(seed: int) -> str:
    """
    Convert an integer seed to a 64-character hex string.

    This is useful for converting legacy integer seeds (e.g., from CLI --seed)
    into the hex format expected by DeterministicPRNG.

    Args:
        seed: Integer seed value.

    Returns:
        64-character lowercase hex string.

    Example:
        >>> int_to_hex_seed(42)
        '000000000000000000000000000000000000000000000000000000000000002a'
    """
    # Use SHA-256 of the integer's canonical representation for uniform distribution
    # This ensures any integer maps to a full 256-bit seed space
    canonical = f"INT_SEED:{seed}".encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class PRNGKey:
    """
    Immutable key for hierarchical seed derivation.

    A PRNGKey consists of:
        - root: The master seed (64-character hex string)
        - path: A tuple of hierarchical labels (e.g., slice, mode, cycle, component)

    The key is frozen (immutable) to ensure it can be safely reused
    without risk of mutation affecting derived seeds.

    Example:
        >>> key = PRNGKey(root="a" * 64, path=("slice_uplift_sparse", "baseline", "cycle_0001"))
        >>> derive_seed(key)
        <deterministic integer>
    """
    root: str  # 64-hex string (SHA-256 digest)
    path: Tuple[str, ...] = ()  # Hierarchical labels

    def __post_init__(self):
        """Validate the key on construction."""
        if len(self.root) != 64:
            raise ValueError(f"root must be 64 hex characters, got {len(self.root)}")
        try:
            int(self.root, 16)
        except ValueError:
            raise ValueError(f"root must be valid hex string, got invalid characters")

    def child(self, *labels: str) -> PRNGKey:
        """
        Derive a child key by appending labels to the path.

        Args:
            *labels: One or more string labels to append.

        Returns:
            New PRNGKey with extended path.

        Example:
            >>> parent = PRNGKey(root="a" * 64, path=("slice",))
            >>> child = parent.child("cycle_0001", "ordering")
            >>> child.path
            ('slice', 'cycle_0001', 'ordering')
        """
        return PRNGKey(root=self.root, path=self.path + labels)

    def canonical_string(self) -> str:
        """
        Generate the canonical string representation for hashing.

        Format: root + "::" + "::".join(path)

        Returns:
            Canonical UTF-8 string.
        """
        if self.path:
            return self.root + "::" + "::".join(self.path)
        return self.root


def derive_seed(key: PRNGKey) -> int:
    """
    Derive a deterministic integer seed from a PRNGKey.

    Algorithm:
        1. Concatenate: root + "::" + "::".join(path) as UTF-8 string.
        2. Compute SHA-256 → 256-bit digest.
        3. Convert first 8 bytes to integer (big-endian).
        4. Return seed % (2**32) for compatibility with random.Random and np.random.

    This function is STATELESS and PURE: the same key always produces
    the same seed, regardless of call order or global state.

    Args:
        key: PRNGKey containing root and path.

    Returns:
        32-bit unsigned integer seed.

    Example:
        >>> key = PRNGKey(root="a" * 64, path=("test",))
        >>> seed1 = derive_seed(key)
        >>> seed2 = derive_seed(key)
        >>> seed1 == seed2
        True
    """
    canonical = key.canonical_string().encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    # First 8 bytes as big-endian unsigned integer
    raw_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    # Reduce to 32-bit for compatibility with standard RNGs
    return raw_seed % (2**32)


def derive_seed_64bit(key: PRNGKey) -> int:
    """
    Derive a 64-bit seed for applications requiring larger seed space.

    Same algorithm as derive_seed but returns the full 64-bit value.

    Args:
        key: PRNGKey containing root and path.

    Returns:
        64-bit unsigned integer seed.
    """
    canonical = key.canonical_string().encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class DeterministicPRNG:
    """
    Hierarchical deterministic PRNG based on SHA-256.

    This class provides a clean interface for obtaining scoped random.Random
    instances from a master seed. Each call to for_path() returns an independent
    RNG instance seeded deterministically from the path.

    IMPORTANT: This class does NOT use global state. It never calls:
        - random.seed()
        - np.random.seed()

    All randomness is isolated to the returned RNG instances.

    Contract Reference:
        Implements the hierarchical seed derivation chain specified in
        docs/DETERMINISM_CONTRACT.md:
            U2_MASTER_SEED → slice_seed → cycle_seed → component_seed

    Example:
        >>> prng = DeterministicPRNG("a" * 64)
        >>> rng = prng.for_path("slice_uplift_sparse", "baseline", "cycle_0001", "ordering")
        >>> values = [rng.random() for _ in range(5)]
        >>> # Re-create with same path → identical sequence
        >>> rng2 = prng.for_path("slice_uplift_sparse", "baseline", "cycle_0001", "ordering")
        >>> values2 = [rng2.random() for _ in range(5)]
        >>> values == values2
        True
    """

    def __init__(self, master_seed: str):
        """
        Initialize with a master seed.

        Args:
            master_seed: 64-character hex string representing the root seed.
                         Can be derived from a manifest hash, config hash, or
                         converted from integer via int_to_hex_seed().

        Raises:
            ValueError: If master_seed is not a valid 64-hex string.
        """
        if len(master_seed) != 64:
            raise ValueError(
                f"master_seed must be 64 hex characters, got {len(master_seed)}"
            )
        try:
            int(master_seed, 16)
        except ValueError:
            raise ValueError("master_seed must be valid hex string")

        self._master_seed = master_seed.lower()
        self._root_key = PRNGKey(root=self._master_seed, path=())

    @property
    def master_seed(self) -> str:
        """Return the master seed (read-only)."""
        return self._master_seed

    def key_for_path(self, *labels: str) -> PRNGKey:
        """
        Create a PRNGKey for the given hierarchical path.

        Args:
            *labels: Hierarchical labels (e.g., slice_name, mode, cycle, component).

        Returns:
            PRNGKey with the specified path.
        """
        return self._root_key.child(*labels)

    def seed_for_path(self, *labels: str) -> int:
        """
        Derive the integer seed for a given path.

        This is useful for logging/auditing seed provenance.

        Args:
            *labels: Hierarchical labels.

        Returns:
            32-bit integer seed.
        """
        return derive_seed(self.key_for_path(*labels))

    def for_path(self, *labels: str) -> random.Random:
        """
        Get a seeded random.Random instance for the given path.

        The returned RNG is independent and isolated. Multiple calls with
        the same labels return RNGs that will produce identical sequences.

        Args:
            *labels: Hierarchical labels (e.g., "slice_uplift_sparse", "baseline",
                     "cycle_0001", "ordering").

        Returns:
            random.Random instance seeded from the derived seed.

        Example:
            >>> prng = DeterministicPRNG("a" * 64)
            >>> rng = prng.for_path("slice", "mode", "cycle_0001", "shuffle")
            >>> shuffled = list(range(10))
            >>> rng.shuffle(shuffled)
        """
        seed = self.seed_for_path(*labels)
        return random.Random(seed)

    def for_numpy(self, *labels: str) -> Any:
        """
        Get a numpy random Generator for the given path.

        Requires numpy to be installed. Returns np.random.Generator using
        the PCG64 bit generator seeded from the derived seed.

        Args:
            *labels: Hierarchical labels.

        Returns:
            numpy.random.Generator instance.

        Raises:
            ImportError: If numpy is not installed.
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for for_numpy()")

        seed = self.seed_for_path(*labels)
        # Use the modern Generator API with PCG64 for better statistical properties
        return np.random.Generator(np.random.PCG64(seed))

    def for_numpy_legacy(self, *labels: str) -> Any:
        """
        Get a numpy RandomState (legacy API) for the given path.

        This is provided for compatibility with code using the older
        np.random.RandomState API.

        Args:
            *labels: Hierarchical labels.

        Returns:
            numpy.random.RandomState instance.

        Raises:
            ImportError: If numpy is not installed.
        """
        if not HAS_NUMPY:
            raise ImportError("numpy is required for for_numpy_legacy()")

        seed = self.seed_for_path(*labels)
        return np.random.RandomState(seed)

    def generate_seed_schedule(
        self,
        num_cycles: int,
        slice_name: str,
        mode: str,
    ) -> List[int]:
        """
        Generate a deterministic seed schedule for experiment cycles.

        This replaces the ad-hoc generate_seed_schedule() function in
        experiments/run_uplift_u2.py with a hierarchically scoped version.

        Each cycle gets its own derived seed based on:
            master_seed → slice_name → mode → f"cycle_{i:04d}" → "seed"

        Args:
            num_cycles: Number of cycles to generate seeds for.
            slice_name: Name of the experiment slice.
            mode: Experiment mode ("baseline" or "rfl").

        Returns:
            List of 32-bit integer seeds, one per cycle.

        Example:
            >>> prng = DeterministicPRNG("a" * 64)
            >>> seeds = prng.generate_seed_schedule(10, "arithmetic_simple", "baseline")
            >>> len(seeds)
            10
        """
        return [
            self.seed_for_path(slice_name, mode, f"cycle_{i:04d}", "seed")
            for i in range(num_cycles)
        ]

    def log_metadata(self, *labels: str) -> dict:
        """
        Generate metadata for logging PRNG state (for audit trail).

        This method produces a dictionary suitable for including in
        telemetry records to enable seed provenance verification.

        Args:
            *labels: Hierarchical labels for the current context.

        Returns:
            Dictionary with PRNG metadata.
        """
        key = self.key_for_path(*labels)
        return {
            "prng_master_seed_prefix": self._master_seed[:16] + "...",
            "prng_path": list(labels),
            "prng_derived_seed": derive_seed(key),
            "prng_canonical_hash": hashlib.sha256(
                key.canonical_string().encode("utf-8")
            ).hexdigest()[:16],
        }


# --- Compatibility shim for legacy code ---

def create_prng_from_int_seed(seed: int) -> DeterministicPRNG:
    """
    Create a DeterministicPRNG from a legacy integer seed.

    This is a convenience function for migrating code that uses integer seeds
    (e.g., from CLI arguments) to the new hierarchical PRNG.

    Args:
        seed: Integer seed value.

    Returns:
        DeterministicPRNG instance.

    Example:
        >>> prng = create_prng_from_int_seed(42)
        >>> isinstance(prng, DeterministicPRNG)
        True
    """
    hex_seed = int_to_hex_seed(seed)
    return DeterministicPRNG(hex_seed)


# --- Self-test when run directly ---

if __name__ == "__main__":
    print("Testing DeterministicPRNG...")

    # Test 1: Seed derivation stability
    key1 = PRNGKey(root="a" * 64, path=("test", "path"))
    key2 = PRNGKey(root="a" * 64, path=("test", "path"))
    assert derive_seed(key1) == derive_seed(key2), "Seed derivation not stable"
    print("✓ Seed derivation is stable")

    # Test 2: Different paths → different seeds
    key_a = PRNGKey(root="a" * 64, path=("path_a",))
    key_b = PRNGKey(root="a" * 64, path=("path_b",))
    assert derive_seed(key_a) != derive_seed(key_b), "Different paths should give different seeds"
    print("✓ Different paths produce different seeds")

    # Test 3: for_path reproducibility
    prng = DeterministicPRNG("b" * 64)
    rng1 = prng.for_path("slice", "mode", "cycle_0001")
    rng2 = prng.for_path("slice", "mode", "cycle_0001")
    seq1 = [rng1.random() for _ in range(10)]
    seq2 = [rng2.random() for _ in range(10)]
    assert seq1 == seq2, "for_path should produce reproducible sequences"
    print("✓ for_path produces reproducible sequences")

    # Test 4: Seed schedule generation
    prng = DeterministicPRNG("c" * 64)
    schedule1 = prng.generate_seed_schedule(5, "test_slice", "baseline")
    schedule2 = prng.generate_seed_schedule(5, "test_slice", "baseline")
    assert schedule1 == schedule2, "Seed schedule should be deterministic"
    assert len(set(schedule1)) == 5, "Seed schedule should have unique seeds per cycle"
    print("✓ Seed schedule generation is deterministic")

    # Test 5: int_to_hex_seed
    hex1 = int_to_hex_seed(42)
    hex2 = int_to_hex_seed(42)
    assert hex1 == hex2, "int_to_hex_seed should be deterministic"
    assert len(hex1) == 64, "Should produce 64-char hex string"
    print("✓ int_to_hex_seed is deterministic")

    # Test 6: No global state pollution
    import random as stdlib_random
    stdlib_state_before = stdlib_random.getstate()
    prng = DeterministicPRNG("d" * 64)
    _ = prng.for_path("test")
    stdlib_state_after = stdlib_random.getstate()
    assert stdlib_state_before == stdlib_state_after, "Should not affect global random state"
    print("✓ No global random state pollution")

    print("\n✅ All DeterministicPRNG tests passed!")

