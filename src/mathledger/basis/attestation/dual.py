"""Dual attestation utilities."""

from __future__ import annotations

import hashlib
from typing import Mapping, Optional, Sequence

from mathledger.basis.core import DualAttestation, HexDigest
from mathledger.basis.crypto.hash import reasoning_root as _reasoning_root
from mathledger.basis.crypto.hash import ui_root as _ui_root


def reasoning_root(events: Sequence[str]) -> HexDigest:
    """Reasoning root wrapper."""
    return _reasoning_root(events)


def ui_root(events: Sequence[str]) -> HexDigest:
    """UI root wrapper."""
    return _ui_root(events)


def composite_root(reasoning: HexDigest, ui: HexDigest) -> HexDigest:
    """Compute SHA256(reasoning || ui)."""
    if len(reasoning) != 64 or len(ui) != 64:
        raise ValueError("Reasoning and UI roots must be 64-char hex digests.")
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
    """Construct dual attestation from event streams."""
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


def verify_attestation(attestation: DualAttestation) -> bool:
    """Recompute and compare composite root."""
    expected = composite_root(attestation.reasoning_root, attestation.ui_root)
    return expected == attestation.composite_root

