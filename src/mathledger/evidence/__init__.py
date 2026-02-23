"""Wave A4 evidence replay primitives."""

from mathledger.evidence.replay import (
    COMPOSITE_FORMULA_NOTE,
    EVIDENCE_SCHEMA_VERSION,
    EvidenceReplayViolation,
    assert_evidence_pack_passes,
    build_evidence_pack,
    compute_h_t,
    compute_r_t_from_artifacts,
    compute_u_t_from_events,
    evidence_pack_json,
    verify_evidence_pack,
)

__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "COMPOSITE_FORMULA_NOTE",
    "EvidenceReplayViolation",
    "compute_u_t_from_events",
    "compute_r_t_from_artifacts",
    "compute_h_t",
    "build_evidence_pack",
    "verify_evidence_pack",
    "assert_evidence_pack_passes",
    "evidence_pack_json",
]

