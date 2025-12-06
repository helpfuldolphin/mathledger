"""Dual attestation orchestration."""

from .dual import (
    attestation_from_block,
    build_attestation,
    composite_root,
    reasoning_root,
    ui_root,
    verify_attestation,
)

__all__ = [
    "attestation_from_block",
    "reasoning_root",
    "ui_root",
    "composite_root",
    "build_attestation",
    "verify_attestation",
]


