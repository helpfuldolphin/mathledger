"""
RFC 8785 (JCS) Canonicalization Substrate.

This module is the single source of truth for JSON canonicalization in MathLedger.
It implements the VCP 2.1 requirement: "All JSON serialization for hashes,
signatures, or ledger storage MUST use RFC 8785 (JCS)."

Vibe:
- Strict.
- No hidden options (indent, sort_keys are implicit in JCS).
- Type-safe.
"""
import hashlib
import jcs  # type: ignore
from typing import Any, Union

# Type alias for JSON-compatible types
JsonType = Union[dict[str, Any], list[Any], str, int, float, bool, None]

def canonical_json_dump(obj: JsonType) -> bytes:
    """
    Serialize an object to canonical JSON bytes per RFC 8785.

    Rules:
    1. Object keys are sorted lexicographically by their UTF-16 code units.
    2. Whitespace is eliminated.
    3. Numbers are represented roughly as per IEEE 754 (see JCS spec).
    4. Unicode characters are unescaped (except for strict control chars).

    Args:
        obj: The object to serialize. Must be a JSON-compatible type.
             Pydantic models MUST be converted to dicts before calling this.

    Returns:
        bytes: The canonical JSON bytes.

    Raises:
        TypeError: If the object is not JSON-serializable.
    """
    return jcs.canonicalize(obj)

def canonical_hash(obj: JsonType) -> str:
    """
    Compute the SHA-256 hash of the canonical JSON representation of an object.

    This is the standard H_t (Hash at time t) operation for MathLedger artifacts.
    It ensures that H(obj) is deterministic across platforms and languages.

    Args:
        obj: The object to hash.

    Returns:
        str: The hex digest of the SHA-256 hash.
    """
    payload = canonical_json_dump(obj)
    return hashlib.sha256(payload).hexdigest()

