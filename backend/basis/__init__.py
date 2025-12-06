"""
The Minimal Basis.

This package contains the orthogonal set of modules that form the
irreducible core of MathLedger.
"""
from .canon import canonical_json_dump, canonical_hash

__all__ = ["canonical_json_dump", "canonical_hash"]

