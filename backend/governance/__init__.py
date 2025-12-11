"""
Governance and provenance validation module.

This module provides cryptographic validation of:
- Governance chains (attestation lineage)
- Declared roots (block Merkle roots)
- Dual-root integrity (R_t, U_t)
- Safety gate decision surfacing (Phase X Neural Link)
"""

from .validator import LawkeeperValidator, GovernanceEntry, DeclaredRoot
from .safety_gate import (
    SafetyEnvelope,
    SafetyGateStatus,
    SafetyGateDecision,
    build_safety_gate_summary_for_first_light,
    build_safety_gate_tile_for_global_health,
    attach_safety_gate_to_evidence,
    build_global_health_surface,
)

__all__ = [
    # Validator
    "LawkeeperValidator",
    "GovernanceEntry",
    "DeclaredRoot",
    # Safety Gate (Phase X)
    "SafetyEnvelope",
    "SafetyGateStatus",
    "SafetyGateDecision",
    "build_safety_gate_summary_for_first_light",
    "build_safety_gate_tile_for_global_health",
    "attach_safety_gate_to_evidence",
    "build_global_health_surface",
]
