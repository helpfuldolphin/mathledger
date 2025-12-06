"""
Deterministic ledger block sealing built on canonical primitives.
"""

from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence

from basis.core import Block, BlockHeader, HexDigest
from basis.crypto.hash import merkle_root
from basis.logic.normalizer import normalize_many


def seal_block(
    statements: Sequence[str],
    *,
    prev_hash: HexDigest,
    block_number: int,
    timestamp: float,
    version: str = "v1",
) -> Block:
    """
    Produce an immutable block with a deterministic Merkle header.

    The statements are normalised and stored as a lexicographically sorted tuple.
    """
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
    """Serialise a block into primitives suitable for JSON output."""
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
    """Return a canonical JSON representation of a block."""
    return json.dumps(block_to_dict(block), sort_keys=True, separators=(",", ":"))

