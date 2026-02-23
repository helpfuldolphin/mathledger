"""Immutable value types for deterministic basis primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


NormalizedFormula = str
HexDigest = str


@dataclass(frozen=True)
class BlockHeader:
    """Deterministic block header."""

    block_number: int
    prev_hash: HexDigest
    merkle_root: HexDigest
    timestamp: float
    version: str = "v1"


@dataclass(frozen=True)
class Block:
    """Immutable block representation."""

    header: BlockHeader
    statements: Tuple[str, ...]


@dataclass(frozen=True)
class DualAttestation:
    """Binds reasoning and UI roots with composite hash."""

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
    """Single tier in the curriculum ladder."""

    identifier: str
    title: str
    description: str
    prerequisites: Tuple[str, ...] = ()
    objectives: Tuple[str, ...] = ()


CurriculumIndex = Dict[str, CurriculumTier]

