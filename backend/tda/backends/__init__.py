"""
TDA Backend Abstractions.

This module provides protocol definitions and implementations for TDA
computation backends (Ripser, GUDHI, fallback).

The TDABackend protocol allows swapping implementations without changing
the monitor code.

Available backends:
- RipserBackend: Uses ripser library (recommended, fast)
- GUDHIBackend: Uses GUDHI library (feature-rich)
- FallbackBackend: Pure Python fallback (limited H_0 only)

References:
    - TDA_MIND_SCANNER_SPEC.md Section 7.1
"""

from backend.tda.backends.protocol import TDABackend

__all__ = ["TDABackend"]
