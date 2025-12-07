"""
Epoch root system for ledger checkpointing.

Provides epoch-level aggregation of block attestation roots.
"""

from .sealer import seal_epoch, compute_epoch_root, DEFAULT_EPOCH_SIZE
from .verifier import verify_epoch_integrity, replay_epoch

__all__ = [
    "seal_epoch",
    "compute_epoch_root",
    "verify_epoch_integrity",
    "replay_epoch",
    "DEFAULT_EPOCH_SIZE",
]
