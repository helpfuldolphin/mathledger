"""Dual attestation orchestration."""

from .dual import (
    build_attestation,
    composite_root,
    reasoning_root,
    ui_root,
    verify_attestation,
)

__all__ = [
    "reasoning_root",
    "ui_root",
    "composite_root",
    "build_attestation",
    "verify_attestation",
]

