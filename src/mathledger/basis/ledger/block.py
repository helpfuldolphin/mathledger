"""Deterministic block sealing primitives."""

from __future__ import annotations

import json
from typing import Mapping, Sequence

from mathledger.basis.core import Block, BlockHeader, HexDigest
from mathledger.basis.crypto.hash import merkle_root
from mathledger.basis.logic.normalizer import normalize_many


def seal_block(
    statements: Sequence[str],
    *,
    prev_hash: HexDigest,
    block_number: int,
    timestamp: float,
    version: str = "v1",
) -> Block:
    """Produce immutable block with deterministic header."""
    normalized = normalize_many(statements)
    merkle = merkle_root(normalized)
    header = BlockHeader(
        block_number=block_number,
        prev_hash=prev_hash,
        merkle_root=merkle,
        timestamp=timestamp,
        version=version,
    )
    return Block(header=header, statements=tuple(sorted(normalized)))


def block_to_dict(block: Block) -> Mapping[str, object]:
    """Serialize block to primitive JSON-safe structure."""
    return {
        "header": {
            "block_number": block.header.block_number,
            "prev_hash": block.header.prev_hash,
            "merkle_root": block.header.merkle_root,
            "timestamp": block.header.timestamp,
            "version": block.header.version,
        },
        "statements": list(block.statements),
    }


def block_json(block: Block) -> str:
    """Canonical compact JSON encoding for deterministic hashing."""
    return json.dumps(block_to_dict(block), sort_keys=True, separators=(",", ":"))

