"""
Manifest verification utilities for uplift_u2 Evidence Packs.

This module provides cryptographic hash computation and verification
functions for attestation artifacts.

PHASE II — NOT USED IN PHASE I
"""

import hashlib
from pathlib import Path
from typing import Optional


def compute_artifact_hash(path: Path) -> str:
    """
    Compute SHA-256 hash of a file artifact.
    
    Args:
        path: Path to the artifact file
        
    Returns:
        SHA-256 hash as lowercase hex string, or "missing" if file doesn't exist
        
    Examples:
        >>> from pathlib import Path
        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        ...     f.write('test content')
        ...     tmp_path = Path(f.name)
        >>> hash_val = compute_artifact_hash(tmp_path)
        >>> len(hash_val)
        64
        >>> tmp_path.unlink()
    """
    if not path.exists():
        return "missing"
    
    sha256_hash = hashlib.sha256()
    
    with open(path, "rb") as f:
        # Read file in chunks for memory efficiency
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()


def verify_artifact_hash(path: Path, expected_hash: str) -> bool:
    """
    Verify that an artifact's hash matches the expected value.
    
    Args:
        path: Path to the artifact file
        expected_hash: Expected SHA-256 hash as hex string
        
    Returns:
        True if hash matches, False otherwise (including if file is missing)
    """
    actual_hash = compute_artifact_hash(path)
    if actual_hash == "missing":
        return False
    return actual_hash == expected_hash


def hash_string(data: str) -> str:
    """
    Compute SHA-256 hash of a string.
    
    Args:
        data: String to hash
        
    Returns:
        SHA-256 hash as lowercase hex string
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()
