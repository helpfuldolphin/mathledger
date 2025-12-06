"""
Deterministic PRNG Implementation for U2 Planner

DESIGN PRINCIPLES:
1. Hierarchical seeding: master_seed -> slice_seed -> operation_seed
2. Platform-independent: uses hashlib for deterministic hashing
3. Serializable state: can be saved/restored for replay
4. Audit trail: all RNG calls are traceable to seed path

USAGE:
    master_prng = DeterministicPRNG("0x1234abcd")
    slice_prng = master_prng.for_path("slice", "arithmetic_simple")
    cycle_prng = slice_prng.for_path("cycle", "42")
    
    # All operations from cycle_prng are deterministic and reproducible
    value = cycle_prng.random()
    choice = cycle_prng.choice([1, 2, 3])
"""

import hashlib
import random
from typing import Any, List, Optional, Sequence, TypeVar

T = TypeVar('T')


def canonicalize_seed(seed: Any) -> str:
    """
    Convert any seed value to canonical hex string format.
    
    Args:
        seed: Integer, hex string, or bytes
        
    Returns:
        Canonical hex string (e.g., "0x1234abcd")
    """
    if isinstance(seed, str):
        if seed.startswith("0x"):
            # Check if it's valid hex after 0x
            hex_part = seed[2:]
            try:
                int(hex_part, 16)
                return seed.lower()
            except ValueError:
                # Not valid hex, hash it
                seed_bytes = seed.encode('utf-8')
                return f"0x{hashlib.sha256(seed_bytes).hexdigest()[:16]}"
        else:
            # Try to interpret as hex without 0x prefix
            try:
                int(seed, 16)
                return f"0x{seed.lower()}"
            except ValueError:
                # Not hex, hash it
                seed_bytes = seed.encode('utf-8')
                return f"0x{hashlib.sha256(seed_bytes).hexdigest()[:16]}"
    elif isinstance(seed, int):
        return f"0x{seed:016x}"
    elif isinstance(seed, bytes):
        return f"0x{seed.hex()}"
    else:
        # Hash arbitrary objects
        seed_bytes = str(seed).encode('utf-8')
        return f"0x{hashlib.sha256(seed_bytes).hexdigest()[:16]}"


def int_to_hex_seed(value: int) -> str:
    """Convert integer to canonical hex seed string."""
    return f"0x{value:016x}"


def hex_to_int_seed(hex_seed: str) -> int:
    """Convert hex seed string to integer."""
    if hex_seed.startswith("0x"):
        hex_seed = hex_seed[2:]
    return int(hex_seed, 16)


class DeterministicPRNG:
    """
    Deterministic pseudo-random number generator with hierarchical seeding.
    
    INVARIANTS:
    - Same seed path always produces same sequence
    - Platform-independent (no reliance on system RNG)
    - State is fully serializable
    - All operations are traceable to seed lineage
    """
    
    def __init__(self, seed: Any, parent_path: Optional[List[str]] = None):
        """
        Initialize PRNG with a seed.
        
        Args:
            seed: Master seed (int, hex string, or bytes)
            parent_path: Hierarchical path from root (for debugging)
        """
        self.seed_canonical = canonicalize_seed(seed)
        self.parent_path = parent_path or []
        self._rng = random.Random(hex_to_int_seed(self.seed_canonical))
        self._call_count = 0
        
    def for_path(self, *path_components: str) -> 'DeterministicPRNG':
        """
        Create a child PRNG for a specific path.
        
        This enables hierarchical isolation:
        - Different slices get independent RNG streams
        - Different cycles within a slice are isolated
        - Deterministic even if execution order changes
        
        Args:
            *path_components: Path elements (e.g., "slice", "arithmetic_simple")
            
        Returns:
            New DeterministicPRNG with derived seed
        """
        # Derive child seed by hashing parent seed + path
        path_str = "/".join(path_components)
        combined = f"{self.seed_canonical}:{path_str}"
        child_seed_bytes = hashlib.sha256(combined.encode('utf-8')).digest()
        child_seed = f"0x{child_seed_bytes.hex()[:16]}"
        
        new_path = self.parent_path + list(path_components)
        return DeterministicPRNG(child_seed, parent_path=new_path)
    
    def random(self) -> float:
        """
        Generate random float in [0.0, 1.0).
        
        Returns:
            Deterministic random float
        """
        self._call_count += 1
        return self._rng.random()
    
    def randint(self, a: int, b: int) -> int:
        """
        Generate random integer in [a, b] inclusive.
        
        Args:
            a: Lower bound (inclusive)
            b: Upper bound (inclusive)
            
        Returns:
            Deterministic random integer
        """
        self._call_count += 1
        return self._rng.randint(a, b)
    
    def choice(self, seq: Sequence[T]) -> T:
        """
        Choose random element from sequence.
        
        Args:
            seq: Non-empty sequence
            
        Returns:
            Deterministic random element
        """
        self._call_count += 1
        return self._rng.choice(seq)
    
    def shuffle(self, seq: List[T]) -> None:
        """
        Shuffle list in-place deterministically.
        
        Args:
            seq: List to shuffle (modified in-place)
        """
        self._call_count += 1
        self._rng.shuffle(seq)
    
    def sample(self, population: Sequence[T], k: int) -> List[T]:
        """
        Sample k elements without replacement.
        
        Args:
            population: Sequence to sample from
            k: Number of elements to sample
            
        Returns:
            List of k sampled elements
        """
        self._call_count += 1
        return self._rng.sample(population, k)
    
    def get_state(self) -> dict:
        """
        Get serializable PRNG state for snapshots.
        
        Returns:
            Dictionary with seed, path, and call count
        """
        rng_state = self._rng.getstate()
        # Convert tuple to list for JSON serialization
        rng_state_serializable = (
            rng_state[0],  # version
            list(rng_state[1]),  # state array (tuple -> list)
            rng_state[2] if len(rng_state) > 2 else None,  # gauss_next
        )
        return {
            "seed": self.seed_canonical,
            "parent_path": self.parent_path,
            "call_count": self._call_count,
            "rng_state": rng_state_serializable,
        }
    
    def set_state(self, state: dict) -> None:
        """
        Restore PRNG state from snapshot.
        
        Args:
            state: State dictionary from get_state()
        """
        self.seed_canonical = state["seed"]
        self.parent_path = state["parent_path"]
        self._call_count = state["call_count"]
        
        # Convert list back to tuple for setstate
        rng_state = state["rng_state"]
        rng_state_tuple = (
            rng_state[0],  # version
            tuple(rng_state[1]),  # state array (list -> tuple)
            rng_state[2] if len(rng_state) > 2 else None,  # gauss_next
        )
        self._rng.setstate(rng_state_tuple)
    
    def get_lineage(self) -> str:
        """
        Get full seed lineage for debugging.
        
        Returns:
            String representation of seed path
        """
        if not self.parent_path:
            return f"root({self.seed_canonical})"
        path_str = "/".join(self.parent_path)
        return f"{path_str}({self.seed_canonical})"
    
    def __repr__(self) -> str:
        return f"DeterministicPRNG({self.get_lineage()}, calls={self._call_count})"
