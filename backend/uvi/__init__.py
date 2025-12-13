"""
User Verified Input (UVI) Module

This module provides the interface for binding human epistemic judgment
to machine reasoning within the MathLedger dual-attestation framework.

CURRENT STATUS: Stub implementation (Phase I)

The UVI module completes the dual-attestation story:
- R_t (Reasoning Root): Machine-generated proof artifacts
- U_t (UI Root): User interaction events including UVI confirmations

Without UVI, the system is model-centric.
With UVI, the system becomes epistemic infrastructure.

SHADOW MODE CONTRACT:
- All UVI events are observational (recorded, not enforced)
- No gating or blocking based on UVI status
- Events flow into U_t Merkle tree for attestation
"""

from backend.uvi.events import (
    UVIEvent,
    UVIEventType,
    record_confirmation,
    record_correction,
    record_flag,
)
from backend.uvi.attestation import (
    compute_uvi_digest,
    build_uvi_attestation_leaf,
)

__all__ = [
    "UVIEvent",
    "UVIEventType",
    "record_confirmation",
    "record_correction",
    "record_flag",
    "compute_uvi_digest",
    "build_uvi_attestation_leaf",
]
