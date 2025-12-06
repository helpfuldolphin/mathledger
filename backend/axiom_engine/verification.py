"""Deprecated verification shim."""

import warnings

from derivation.verification import *  # noqa: F401,F403

warnings.warn(
    "backend.axiom_engine.verification is deprecated; import derivation.verification instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["VerificationOutcome", "LeanFallback", "StatementVerifier"]
