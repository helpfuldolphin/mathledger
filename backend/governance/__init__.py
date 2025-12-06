"""
Governance and provenance validation module.

This module provides cryptographic validation of:
- Governance chains (attestation lineage)
- Declared roots (block Merkle roots)
- Dual-root integrity (R_t, U_t)
"""

from .validator import LawkeeperValidator, GovernanceEntry, DeclaredRoot

__all__ = ["LawkeeperValidator", "GovernanceEntry", "DeclaredRoot"]
