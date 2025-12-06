"""
Lean project accessors for MathLedger substrate.

Provides utilities to locate the Lean build artifacts from Python code
without relying on the legacy backend path.
"""

from pathlib import Path

LEAN_PROJECT_ROOT = Path(__file__).resolve().parent
LAKEFILE = LEAN_PROJECT_ROOT / "lakefile.lean"

__all__ = ["LEAN_PROJECT_ROOT", "LAKEFILE"]

