"""
Shared typed structures for the canonical MathLedger basis.

These dataclasses are immutable, hashable value types that capture the
deterministic state passed between subsystems. They enforce explicit data
requirements at module boundaries – no implicit globals or ambient context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple, Dict, Any


NormalizedFormula = str
HexDigest = str


@dataclass(frozen=True)
class BlockHeader:
    """Deterministic ledger header describing a sealed block."""

    block_number: int
    prev_hash: HexDigest
    merkle_root: HexDigest
    timestamp: float
    version: str = "v1"


@dataclass(frozen=True)
class Block:
    """Immutable representation of a sealed ledger block."""

    header: BlockHeader
    statements: Tuple[str, ...]


@dataclass(frozen=True)
class DualAttestation:
    """
    Dual attestation binding reasoning and UI roots.

    The composite root must be SHA256(reasoning_root || ui_root) in ASCII.
    """

    reasoning_root: HexDigest
    ui_root: HexDigest
    composite_root: HexDigest
    reasoning_event_count: int = 0
    ui_event_count: int = 0
    version: str = "v1"
    algorithm: str = "SHA256"
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CurriculumTier:
    """Single rung on the curriculum ladder."""

    identifier: str
    title: str
    description: str
    prerequisites: Tuple[str, ...] = ()
    objectives: Tuple[str, ...] = ()


CurriculumIndex = Dict[str, CurriculumTier]


