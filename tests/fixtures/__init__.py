# tests/fixtures/__init__.py
"""
Shared test fixtures for MathLedger.

This module provides canonical fixtures that are shared across unit,
integration, and basis tests to ensure deterministic reproducibility.
"""

from .first_organism import (
    CANONICAL_FIRST_ORGANISM_ATTESTATION,
    load_first_organism_attestation,
    make_attested_run_context,
    FirstOrganismFixture,
)

__all__ = [
    "CANONICAL_FIRST_ORGANISM_ATTESTATION",
    "load_first_organism_attestation",
    "make_attested_run_context",
    "FirstOrganismFixture",
]

