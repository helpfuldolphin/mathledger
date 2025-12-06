# PHASE II — NOT USED IN PHASE I
# Deterministic PRNG implementation for reproducible experiments

import hashlib
import random
from typing import Any


def int_to_hex_seed(seed: int) -> str:
    """Convert integer seed to hex string."""
    return f"{seed:016x}"


class DeterministicPRNG:
    """
    Deterministic pseudo-random number generator for reproducible experiments.
    
    Supports hierarchical seeding via for_path() to create isolated PRNG contexts
    for different subsystems while maintaining determinism.
    """
    
    def __init__(self, seed: str):
        """
        Initialize with a hex seed string.
        
        Args:
            seed: Hex string seed for deterministic generation
        """
        self.seed = seed
        self._rng = random.Random(seed)
    
    def for_path(self, *path_components: Any) -> "DeterministicPRNG":
        """
        Create a child PRNG with deterministic seed derived from path.
        
        This allows hierarchical isolation of randomness while maintaining
        determinism across the entire experiment.
        
        Args:
            *path_components: Path components to create child context
            
        Returns:
            New DeterministicPRNG instance with derived seed
        """
        path_str = "/".join(str(c) for c in path_components)
        combined = f"{self.seed}/{path_str}"
        child_seed = hashlib.sha256(combined.encode()).hexdigest()[:16]
        return DeterministicPRNG(child_seed)
    
    def random(self) -> float:
        """Generate random float in [0, 1)."""
        return self._rng.random()
    
    def randint(self, a: int, b: int) -> int:
        """Generate random integer in [a, b]."""
        return self._rng.randint(a, b)
    
    def shuffle(self, x: list) -> None:
        """Shuffle list in-place."""
        self._rng.shuffle(x)
    
    def choice(self, seq):
        """Choose random element from sequence."""
        return self._rng.choice(seq)
