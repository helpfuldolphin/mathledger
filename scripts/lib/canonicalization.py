"""
Shared canonicalization utilities for MathLedger governance.

This module provides RFC 8785 (JSON Canonicalization Scheme) compliant
canonicalization for all governance artifacts.
"""

from typing import Any

try:
    import canonicaljson
    HAS_CANONICALJSON = True
except ImportError:
    import json
    HAS_CANONICALJSON = False


def canonicalize_json(obj: Any) -> str:
    """
    Serialize an object to RFC 8785 canonical JSON.
    
    This function uses the `canonicaljson` library for full RFC 8785 compliance
    if available. If the library is not installed, it falls back to a simplified
    implementation using the standard library.
    
    Args:
        obj: A JSON-serializable Python object.
    
    Returns:
        A UTF-8 string containing the canonical JSON representation.
    
    Raises:
        TypeError: If the object is not JSON-serializable.
    """
    if HAS_CANONICALJSON:
        # Use canonicaljson for full RFC 8785 compliance
        canonical_bytes = canonicaljson.encode_canonical_json(obj)
        return canonical_bytes.decode('utf-8')
    else:
        # Fallback to simplified implementation
        # This handles most cases but does not provide full RFC 8785 compliance
        # (e.g., Unicode normalization, precise number formatting)
        return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(',', ':'))
