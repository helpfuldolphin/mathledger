#!/usr/bin/env python3
"""
Governance Commitment Registry Hashing
=======================================

Provides deterministic hashing of the governance commitment registry
for inclusion in evidence pack manifests.

The registry hash binds a run to a specific governance commitment set,
enabling verification that the run operated under declared constraints.

Usage:
    from governance.registry_hash import compute_registry_hash, load_registry

    hash_value = compute_registry_hash()
    registry = load_registry()
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

# Registry file path (relative to repo root)
DEFAULT_REGISTRY_PATH = Path(__file__).parent / "commitment_registry.json"


def canonicalize_json(data: Any) -> str:
    """
    Canonicalize JSON data for deterministic hashing.

    Uses canonical JSON serialization (RFC 8785-style, not fully compliant):
    - Keys sorted lexicographically at all nesting levels
    - No whitespace between tokens (compact form)
    - ASCII-safe encoding (non-ASCII escaped as \\uXXXX)
    - UTF-8 output encoding

    Note: This is NOT fully RFC 8785 compliant. RFC 8785 specifies additional
    number formatting rules (no trailing zeros, specific exponent handling)
    that Python's json.dumps does not guarantee. For this registry (which
    contains only strings and simple objects), the difference is immaterial.

    Args:
        data: JSON-serializable data

    Returns:
        Canonical JSON string
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_registry_hash(registry_path: Optional[Path] = None) -> str:
    """
    Compute SHA-256 hash of the governance commitment registry.

    Args:
        registry_path: Path to registry JSON (default: governance/commitment_registry.json)

    Returns:
        64-character hexadecimal hash string

    Raises:
        FileNotFoundError: If registry file does not exist
        json.JSONDecodeError: If registry is not valid JSON
    """
    path = registry_path or DEFAULT_REGISTRY_PATH

    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    content = json.loads(path.read_text(encoding="utf-8"))
    canonical = canonicalize_json(content)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_registry(registry_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load the governance commitment registry.

    Args:
        registry_path: Path to registry JSON (default: governance/commitment_registry.json)

    Returns:
        Registry dictionary

    Raises:
        FileNotFoundError: If registry file does not exist
        json.JSONDecodeError: If registry is not valid JSON
    """
    path = registry_path or DEFAULT_REGISTRY_PATH

    if not path.exists():
        raise FileNotFoundError(f"Registry file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_registry_version(registry_path: Optional[Path] = None) -> str:
    """
    Get the schema version from the registry.

    Args:
        registry_path: Path to registry JSON

    Returns:
        Schema version string
    """
    registry = load_registry(registry_path)
    return registry.get("schema_version", "unknown")


if __name__ == "__main__":
    # Self-test
    print("Computing registry hash...")
    hash_value = compute_registry_hash()
    version = get_registry_version()
    print(f"Registry version: {version}")
    print(f"Registry SHA-256: {hash_value}")
