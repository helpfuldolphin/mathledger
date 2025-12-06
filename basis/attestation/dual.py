"""
Dual attestation primitives binding reasoning and UI event streams.
"""

from __future__ import annotations

import hashlib
from typing import Mapping, Optional, Sequence

from basis.core import Block, DualAttestation, HexDigest
from basis.crypto.hash import reasoning_root as _reasoning_root
from basis.crypto.hash import ui_root as _ui_root


def reasoning_root(events: Sequence[str]) -> HexDigest:
    """Public wrapper to emphasise intent."""
    return _reasoning_root(events)


def ui_root(events: Sequence[str]) -> HexDigest:
    """Public wrapper to emphasise intent."""
    return _ui_root(events)


def composite_root(reasoning: HexDigest, ui: HexDigest) -> HexDigest:
    """Compute SHA256(reasoning || ui) with validation."""
    if len(reasoning) != 64 or len(ui) != 64:
        raise ValueError("Reasoning and UI roots must be 64-character hex digests.")
    int(reasoning, 16)
    int(ui, 16)
    payload = f"{reasoning}{ui}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_attestation(
    *,
    reasoning_events: Sequence[str],
    ui_events: Sequence[str],
    extra: Optional[Mapping[str, object]] = None,
) -> DualAttestation:
    """Construct a `DualAttestation` from event streams."""
    r_root = reasoning_root(reasoning_events)
    u_root = ui_root(ui_events)
    h_root = composite_root(r_root, u_root)
    return DualAttestation(
        reasoning_root=r_root,
        ui_root=u_root,
        composite_root=h_root,
        reasoning_event_count=len(reasoning_events),
        ui_event_count=len(ui_events),
        extra=dict(extra) if extra else {},
    )


def attestation_from_block(block: Block, ui_events: Sequence[str]) -> DualAttestation:
    """
    Convenience builder for the First Organism flow.

    Uses the normalized statement list already stored in `Block.statements`
    as the reasoning events, so caller can base the attestation directly on
    the sealed block and the UI event stream without recomputing hashes.
    """
    return build_attestation(reasoning_events=block.statements, ui_events=ui_events)


def verify_attestation(attestation: DualAttestation) -> bool:
    """Return True when attestation hashes recompute correctly."""
    expected = composite_root(attestation.reasoning_root, attestation.ui_root)
    return expected == attestation.composite_root

