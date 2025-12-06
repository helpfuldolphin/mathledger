"""
Deterministic PRNG — Stub Implementation for Testing

This is a STUB implementation of the DeterministicPRNG referenced in
run_uplift_u2.py. The real implementation should be provided by the
project maintainers.

This stub provides enough functionality to test the noise model.

Author: Manus-C (Telemetry Architect)
Date: 2025-12-06
"""

import hashlib
import random
from typing import Optional


def int_to_hex_seed(seed: int) -> str:
    """Convert integer seed to hex string.
    
    Args:
        seed: Integer seed
    
    Returns:
        Hex string representation
    """
    return f"{seed:016x}"


class DeterministicPRNG:
    """Deterministic pseudo-random number generator with hierarchical seeding.
    
    This PRNG ensures:
    - Identical seeds produce identical sequences
    - Hierarchical path-based seeding for independent streams
    - Full reproducibility across runs
    
    NOTE: This is a STUB implementation. The real implementation should
    use a cryptographically secure PRNG or a well-tested library.
    """
    
    def __init__(self, seed: str):
        """Initialize PRNG with hex seed string.
        
        Args:
            seed: Hex string seed (e.g., from int_to_hex_seed)
        """
        self.seed = seed
        self._rng = random.Random(seed)
    
    def for_path(self, *path_components: str) -> "DeterministicPRNG":
        """Create a child PRNG for a specific path.
        
        This enables hierarchical seeding: each path gets an independent
        but deterministic random stream.
        
        Args:
            *path_components: Path components (e.g., "timeout", "context_1")
        
        Returns:
            New DeterministicPRNG instance for this path
        """
        # Derive child seed from parent seed and path
        path_str = "/".join(path_components)
        combined = f"{self.seed}::{path_str}"
        child_seed = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
        return DeterministicPRNG(child_seed)
    
    def random(self) -> float:
        """Generate random float in [0, 1).
        
        Returns:
            Random float
        """
        return self._rng.random()
    
    def uniform(self, a: float, b: float) -> float:
        """Generate random float in [a, b].
        
        Args:
            a: Lower bound
            b: Upper bound
        
        Returns:
            Random float in [a, b]
        """
        return self._rng.uniform(a, b)
    
    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b].
        
        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)
        
        Returns:
            Random integer
        """
        return self._rng.randint(a, b)
    
    def choice(self, seq):
        """Choose random element from sequence.
        
        Args:
            seq: Sequence to choose from
        
        Returns:
            Random element
        """
        return self._rng.choice(seq)
    
    def shuffle(self, seq):
        """Shuffle sequence in-place.
        
        Args:
            seq: Sequence to shuffle
        """
        self._rng.shuffle(seq)
