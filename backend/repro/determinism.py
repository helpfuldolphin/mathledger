#!/usr/bin/env python3
"""
Determinism enforcement helpers for MathLedger.

This module provides utilities to eliminate nondeterminism from:
- Timestamps (use fixed epoch or deterministic seeds)
- Random number generation (seeded RNG)
- UUID generation (deterministic UUIDs from content hashes)
- Dictionary/set iteration order (sorted keys)

Usage:
    from backend.repro.determinism import (
        deterministic_timestamp,
        deterministic_uuid,
        seeded_rng,
        sorted_dict_items
    )
"""

import hashlib
import json
import uuid
import random
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Tuple

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False



DETERMINISTIC_EPOCH = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
"""Fixed epoch for deterministic timestamps."""


def deterministic_timestamp(seed: int = 0) -> datetime:
    """
    Generate a deterministic timestamp from a seed.
    
    Args:
        seed: Integer seed (default 0 for fixed epoch)
    
    Returns:
        Deterministic datetime object
    
    Example:
        >>> ts1 = deterministic_timestamp(42)
        >>> ts2 = deterministic_timestamp(42)
        >>> ts1 == ts2
        True
    """
    if seed == 0:
        return DETERMINISTIC_EPOCH
    
    offset_seconds = (seed % (365 * 24 * 3600))  # Within one year
    return datetime.fromtimestamp(
        DETERMINISTIC_EPOCH.timestamp() + offset_seconds,
        tz=timezone.utc
    )


def deterministic_unix_timestamp(seed: int = 0) -> int:
    """
    Generate a deterministic Unix timestamp from a seed.
    
    Args:
        seed: Integer seed (default 0 for fixed epoch)
    
    Returns:
        Deterministic Unix timestamp (integer seconds since epoch)
    """
    return int(deterministic_timestamp(seed).timestamp())



def deterministic_uuid(content: str, namespace: str = "mathledger") -> str:
    """
    Generate a deterministic UUID from content.
    
    Uses UUID v5 (SHA-1 based) with a fixed namespace to ensure
    identical content always produces the same UUID.
    
    Args:
        content: Content to hash (e.g., normalized statement)
        namespace: Namespace string (default "mathledger")
    
    Returns:
        Deterministic UUID string
    
    Example:
        >>> uuid1 = deterministic_uuid("p->p")
        >>> uuid2 = deterministic_uuid("p->p")
        >>> uuid1 == uuid2
        True
    """
    namespace_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, namespace)
    
    return str(uuid.uuid5(namespace_uuid, content))


def deterministic_uuid_from_hash(hash_hex: str) -> str:
    """
    Generate a deterministic UUID from a hash string.
    
    Args:
        hash_hex: Hexadecimal hash string (e.g., SHA-256)
    
    Returns:
        Deterministic UUID string
    
    Example:
        >>> uuid1 = deterministic_uuid_from_hash("abc123")
        >>> uuid2 = deterministic_uuid_from_hash("abc123")
        >>> uuid1 == uuid2
        True
    """
    hash_prefix = hash_hex[:32].ljust(32, '0')
    return f"{hash_prefix[:8]}-{hash_prefix[8:12]}-{hash_prefix[12:16]}-{hash_prefix[16:20]}-{hash_prefix[20:32]}"



class SeededRNG:
    """
    Seeded random number generator for deterministic randomness.
    
    Example:
        >>> rng1 = SeededRNG(42)
        >>> rng2 = SeededRNG(42)
        >>> rng1.random() == rng2.random()
        True
    """
    
    def __init__(self, seed: int):
        """Initialize with a seed."""
        self.seed = seed
        if HAS_NUMPY:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = random.Random(seed)
    
    def random(self, size: int = 1):
        """Generate random floats in [0, 1)."""
        if HAS_NUMPY:
            return self.rng.random(size)
        else:
            return [self.rng.random() for _ in range(size)]
    
    def randint(self, low: int, high: int, size: int = 1):
        """Generate random integers in [low, high)."""
        if HAS_NUMPY:
            return self.rng.randint(low, high, size)
        else:
            return [self.rng.randint(low, high - 1) for _ in range(size)]
    
    def choice(self, arr: List[Any], size: int = 1, replace: bool = True):
        """Choose random elements from array."""
        if HAS_NUMPY:
            return self.rng.choice(arr, size, replace=replace)
        else:
            if replace:
                return [self.rng.choice(arr) for _ in range(size)]
            else:
                return self.rng.sample(arr, size)
    
    def shuffle(self, arr: List[Any]) -> List[Any]:
        """Shuffle array in-place deterministically."""
        arr_copy = arr.copy()
        if HAS_NUMPY:
            self.rng.shuffle(arr_copy)
        else:
            self.rng.shuffle(arr_copy)
        return arr_copy


def seeded_rng(seed: int) -> SeededRNG:
    """
    Create a seeded RNG instance.
    
    Args:
        seed: Integer seed
    
    Returns:
        SeededRNG instance
    """
    return SeededRNG(seed)



def sorted_dict_items(d: Dict[Any, Any]) -> List[Tuple[Any, Any]]:
    """
    Return dictionary items in sorted order for deterministic iteration.
    
    Args:
        d: Dictionary to sort
    
    Returns:
        List of (key, value) tuples in sorted order
    
    Example:
        >>> d = {'z': 1, 'a': 2, 'm': 3}
        >>> sorted_dict_items(d)
        [('a', 2), ('m', 3), ('z', 1)]
    """
    return sorted(d.items(), key=lambda x: str(x[0]))


def sorted_dict_keys(d: Dict[Any, Any]) -> List[Any]:
    """
    Return dictionary keys in sorted order.
    
    Args:
        d: Dictionary
    
    Returns:
        List of keys in sorted order
    """
    return sorted(d.keys(), key=str)


def sorted_set(s: set) -> List[Any]:
    """
    Return set elements in sorted order.
    
    Args:
        s: Set to sort
    
    Returns:
        List of elements in sorted order
    """
    return sorted(s, key=str)



def _normalize_part(part: Any) -> bytes:
    """
    Normalize input into canonical byte representation.
    
    Args:
        part: Input data (bytes, str, JSON-serializable, or other)
    
    Returns:
        Bytes suitable for hashing.
    """
    if isinstance(part, bytes):
        return part
    if isinstance(part, str):
        return part.encode("utf-8")
    
    json_serializable_types = (dict, list, tuple, int, float, bool, type(None))
    
    if isinstance(part, json_serializable_types):
        try:
            canonical_json = json.dumps(
                part,
                ensure_ascii=True,
                separators=(',', ':'),
                sort_keys=True
            )
            return canonical_json.encode("ascii")
        except (TypeError, ValueError):
            pass
    
    return repr(part).encode("utf-8")


def _canonicalize_parts(parts: Tuple[Any, ...]) -> bytes:
    """Join normalized parts with a sentinel to produce canonical payload."""
    if not parts:
        return b""
    normalized = [_normalize_part(part) for part in parts]
    return b"\x1f".join(normalized)


def deterministic_hash(content: Any, algorithm: str = "sha256") -> str:
    """
    Compute deterministic hash of content.
    
    Args:
        content: Content to hash (bytes, str, or JSON-serializable)
        algorithm: Hash algorithm (default "sha256")
    
    Returns:
        Hexadecimal hash string
    
    Example:
        >>> h1 = deterministic_hash("p->p")
        >>> h2 = deterministic_hash("p->p")
        >>> h1 == h2
        True
    """
    payload = _normalize_part(content)
    
    h = hashlib.new(algorithm)
    h.update(payload)
    return h.hexdigest()


def deterministic_seed_from_content(*parts: Any, algorithm: str = "sha256") -> int:
    """
    Derive a deterministic seed from arbitrary content.

    Args:
        *parts: Parts to hash together (bytes, str, or JSON-serializable).
        algorithm: Hash algorithm (default "sha256").

    Returns:
        64-bit integer seed.
    """
    payload = _canonicalize_parts(parts)
    digest = hashlib.new(algorithm, payload).hexdigest()
    return int(digest[:16], 16)


def deterministic_timestamp_from_content(*parts: Any) -> datetime:
    """
    Produce a deterministic timestamp derived from content.

    Args:
        *parts: Parts to hash together.

    Returns:
        Deterministic datetime.
    """
    seed = deterministic_seed_from_content(*parts)
    return deterministic_timestamp(seed)


def deterministic_isoformat(*parts: Any, resolution: str = "seconds") -> str:
    """
    Serialize deterministic timestamp as ISO-8601 string.

    Args:
        *parts: Parts to hash together.
        resolution: "seconds" (default) or "milliseconds".

    Returns:
        ISO-8601 timestamp string.
    """
    ts = deterministic_timestamp_from_content(*parts)
    if resolution == "seconds":
        ts = ts.replace(microsecond=0)
    elif resolution == "milliseconds":
        ts = ts.replace(microsecond=(ts.microsecond // 1000) * 1000)
    return ts.isoformat()


def deterministic_run_id(prefix: str, *parts: Any, length: int = 12) -> str:
    """
    Construct a deterministic run identifier.

    Args:
        prefix: Prefix string.
        *parts: Parts to hash.
        length: Length of hash suffix (default 12).

    Returns:
        Run identifier.
    """
    digest = deterministic_hash(_canonicalize_parts(parts))
    return f"{prefix}-{digest[:length]}"


def deterministic_slug(*parts: Any, length: int = 24) -> str:
    """
    Produce filesystem-safe deterministic slug.

    Args:
        *parts: Parts to hash.
        length: Length of slug (default 24).

    Returns:
        Slug string.
    """
    digest = deterministic_hash(_canonicalize_parts(parts))
    return digest[:length]


def deterministic_merkle_root(hashes: List[str]) -> str:
    """
    Compute deterministic Merkle root from list of hashes.
    
    Sorts hashes before computing to ensure order-independence.
    
    Args:
        hashes: List of hash strings
    
    Returns:
        Merkle root hash
    
    Example:
        >>> root1 = deterministic_merkle_root(["abc", "def", "ghi"])
        >>> root2 = deterministic_merkle_root(["ghi", "abc", "def"])
        >>> root1 == root2
        True
    """
    if not hashes:
        return deterministic_hash("")
    
    sorted_hashes = sorted(hashes)
    
    level = [deterministic_hash(h) for h in sorted_hashes]
    
    while len(level) > 1:
        next_level = []
        for i in range(0, len(level), 2):
            if i + 1 < len(level):
                combined = level[i] + level[i + 1]
            else:
                combined = level[i] + level[i]  # Duplicate last element
            next_level.append(deterministic_hash(combined))
        level = next_level
    
    return level[0]



def verify_determinism(func, *args, runs: int = 10, **kwargs) -> bool:
    """
    Verify that a function produces deterministic output.
    
    Args:
        func: Function to test
        *args: Positional arguments to func
        runs: Number of test runs (default 10)
        **kwargs: Keyword arguments to func
    
    Returns:
        True if all runs produce identical output
    
    Example:
        >>> def deterministic_func(x):
        ...     return x * 2
        >>> verify_determinism(deterministic_func, 5)
        True
    """
    results = []
    for _ in range(runs):
        result = func(*args, **kwargs)
        results.append(result)
    
    first = results[0]
    return all(r == first for r in results)


if __name__ == "__main__":
    print("Testing determinism helpers...")
    
    ts1 = deterministic_timestamp(42)
    ts2 = deterministic_timestamp(42)
    assert ts1 == ts2, "Timestamp determinism failed"
    print("✓ Timestamps are deterministic")
    
    uuid1 = deterministic_uuid("p->p")
    uuid2 = deterministic_uuid("p->p")
    assert uuid1 == uuid2, "UUID determinism failed"
    print("✓ UUIDs are deterministic")
    
    rng1 = seeded_rng(42)
    rng2 = seeded_rng(42)
    r1 = rng1.random(5)
    r2 = rng2.random(5)
    if HAS_NUMPY:
        assert np.array_equal(r1, r2), "RNG determinism failed"
    else:
        assert r1 == r2, "RNG determinism failed"
    print("✓ RNG is deterministic")
    
    root1 = deterministic_merkle_root(["abc", "def", "ghi"])
    root2 = deterministic_merkle_root(["ghi", "abc", "def"])
    assert root1 == root2, "Merkle root determinism failed"
    print("✓ Merkle roots are deterministic")
    
    print("\nAll determinism tests passed!")
