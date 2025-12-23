"""
Manifest Verifier - SHA-256 Utilities for Attestation Auditing
================================================================

Provides deterministic hash calculation and manifest integrity verification
for experiment attestation artifacts. All operations are read-only.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional


def compute_sha256_file(filepath: Path) -> Optional[str]:
    """
    Compute SHA-256 hash of a file.
    
    Args:
        filepath: Path to the file to hash
        
    Returns:
        Hexadecimal SHA-256 hash string, or None if file doesn't exist
    """
    if not filepath.exists():
        return None
    
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_sha256_string(data: str) -> str:
    """
    Compute SHA-256 hash of a string.
    
    Args:
        data: String data to hash
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """
    Compute SHA-256 hash of bytes.
    
    Args:
        data: Bytes data to hash
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    return hashlib.sha256(data).hexdigest()


def compute_sha256_json(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of JSON data with canonical serialization.
    
    Args:
        data: Dictionary to hash
        
    Returns:
        Hexadecimal SHA-256 hash string
    """
    # Use sorted, compact JSON for deterministic hashing
    canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
    return compute_sha256_string(canonical)


def verify_manifest_file_hash(manifest_path: Path, expected_hash: Optional[str] = None) -> Dict[str, Any]:
    """
    Verify a manifest file exists and optionally check its hash.
    
    Args:
        manifest_path: Path to manifest file
        expected_hash: Optional expected SHA-256 hash to verify
        
    Returns:
        Dictionary with verification results:
        - exists: bool
        - actual_hash: str or None
        - hash_match: bool or None
        - error: str or None
    """
    result = {
        "exists": manifest_path.exists(),
        "actual_hash": None,
        "hash_match": None,
        "error": None
    }
    
    if not result["exists"]:
        result["error"] = "Manifest file not found"
        return result
    
    try:
        result["actual_hash"] = compute_sha256_file(manifest_path)
        
        if expected_hash:
            result["hash_match"] = (result["actual_hash"] == expected_hash)
            if not result["hash_match"]:
                result["error"] = f"Hash mismatch: expected {expected_hash}, got {result['actual_hash']}"
    except Exception as e:
        result["error"] = f"Failed to compute hash: {str(e)}"
    
    return result


def load_and_verify_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file and verify it's valid.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary with:
        - valid: bool
        - data: dict or None
        - error: str or None
        - sha256: str or None
    """
    result = {
        "valid": False,
        "data": None,
        "error": None,
        "sha256": None
    }
    
    if not filepath.exists():
        result["error"] = "File not found"
        return result
    
    try:
        with open(filepath, 'r') as f:
            result["data"] = json.load(f)
        result["valid"] = True
        result["sha256"] = compute_sha256_file(filepath)
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON: {str(e)}"
    except Exception as e:
        result["error"] = f"Failed to load file: {str(e)}"
    
    return result
